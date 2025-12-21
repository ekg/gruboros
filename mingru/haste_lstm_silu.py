"""Fused Haste LSTM + SiLU - uses the new fused CUDA kernel.

This uses the new LSTM_SiLU from haste which has the selectivity gate fused
into the CUDA kernel. Features:
1. Supports BF16 natively
2. Has the silu gate fused into the kernel
3. LSTM has cell state for longer memory + gating for gradient flow

Architecture (all fused in CUDA):
1. LSTM recurrence: c, h = LSTM(x, c_prev, h_prev)
2. SiLU output gate: output = h * silu(Wg_x @ x + Wg_h @ h + bg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import LSTM_SiLU
    HASTE_LSTM_SILU_AVAILABLE = True
except ImportError:
    HASTE_LSTM_SILU_AVAILABLE = False
    print("Warning: haste_pytorch LSTM_SiLU not available.")


class RMSNorm(nn.Module):
    """RMSNorm for stabilizing outputs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


class HasteLSTMSilu(nn.Module):
    """
    Fused Haste LSTM + SiLU using the new fused CUDA kernel.

    LSTM vs GRU:
    - LSTM has separate cell state (c) for memory, hidden state (h) for output
    - LSTM gates: input (i), forget (f), output (o), cell (g)
    - Potentially better long-range memory than GRU

    Key advantages:
    - BF16 native support
    - Fused silu gate in CUDA kernel
    - LSTM memory + selectivity
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        **kwargs
    ):
        super().__init__()
        if not HASTE_LSTM_SILU_AVAILABLE:
            raise RuntimeError("haste_pytorch LSTM_SiLU required. Rebuild haste with LSTM_SiLU support.")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Fused LSTM + SiLU - supports BF16 natively!
        self.lstm_silu = LSTM_SiLU(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            batch_first=False,  # Time-first for haste
        )

        # Normalization after gated output
        self.post_gate_norm = RMSNorm(self.dim_inner)

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize output projection small for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def _lstm_forward(self, x_proj, state):
        """Helper function for gradient checkpointing."""
        output, (h_final, c_final) = self.lstm_silu(x_proj, state=state)
        return output, h_final, c_final

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: (batch, seq_len, dim) - batch-first input
            prev_hiddens: Tuple of (h, c) or None
                h: (batch, dim_inner) hidden state
                c: (batch, dim_inner) cell state
        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: Tuple (h, c) if return_hiddens else None
            next_conv_buffers: None (no conv)
        """
        batch, seq_len, _ = x.shape
        orig_dtype = x.dtype

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Convert to time-first format for haste: (seq_len, batch, dim_inner)
        x_proj_t = x_proj.transpose(0, 1).contiguous()

        # Prepare initial state - LSTM needs both h and c
        # haste LSTM_SiLU expects state as tuple ((1, batch, hidden), (1, batch, hidden))
        if prev_hiddens is not None:
            h0, c0 = prev_hiddens
            h0 = h0.unsqueeze(0)  # (1, batch, dim_inner)
            c0 = c0.unsqueeze(0)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=orig_dtype)
            c0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=orig_dtype)
        state = (h0, c0)

        # CHUNKED PROCESSING for gradient checkpointing
        chunk_size = self.recurrence_chunk_size
        lstm_outputs = []

        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            x_chunk = x_proj_t[chunk_start:chunk_end]

            if self.use_gradient_checkpointing and self.training:
                chunk_out, h, c = checkpoint(self._lstm_forward, x_chunk, state, use_reentrant=False)
                state = (h.unsqueeze(0), c.unsqueeze(0))
            else:
                chunk_out, (h, c) = self.lstm_silu(x_chunk, state=state)
                state = (h, c)

            lstm_outputs.append(chunk_out)

        # Concatenate all chunk outputs
        lstm_out = torch.cat(lstm_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first: (batch, seq_len, dim_inner)
        lstm_out = lstm_out.transpose(0, 1).contiguous()

        # Normalize after gating
        lstm_out = self.post_gate_norm(lstm_out)

        # Project output
        output = self.output_proj(lstm_out)  # (batch, seq_len, dim)

        if return_hiddens:
            # Extract final states
            h_final = state[0].squeeze(0)  # (batch, dim_inner)
            c_final = state[1].squeeze(0)  # (batch, dim_inner)
            return output, (h_final, c_final), None
        else:
            return output

    def __repr__(self):
        return f"HasteLSTMSilu(dim={self.dim}, dim_inner={self.dim_inner}, BF16 native)"

"""Fused Haste GRU + SiLU - uses the new fused CUDA kernel.

This uses the new GRU_SiLU from haste which has the selectivity gate fused
into the CUDA kernel for maximum performance. Unlike the original HasteGRUSilu,
this version:
1. Supports BF16 natively (no fp32 conversion needed)
2. Has the silu gate fused into the kernel (faster)
3. Still has proper GRU skip connection (z*h + (1-z)*g)

Architecture (all fused in CUDA):
1. GRU recurrence with proper skip connection
2. SiLU output gate: output = h * silu(Wg_x @ x + Wg_h @ h + bg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import GRU_SiLU
    HASTE_GRU_SILU_AVAILABLE = True
except ImportError:
    HASTE_GRU_SILU_AVAILABLE = False
    print("Warning: haste_pytorch GRU_SiLU not available.")


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


class HasteGRUSiluFused(nn.Module):
    """
    Fused Haste GRU + SiLU using the new fused CUDA kernel.

    Key advantages over HasteGRUSilu:
    - BF16 native support (no fp32 conversion overhead)
    - Fused silu gate in CUDA kernel (faster)
    - Same GRU skip connection for gradient flow
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
        if not HASTE_GRU_SILU_AVAILABLE:
            raise RuntimeError("haste_pytorch GRU_SiLU required. Rebuild haste with GRU_SiLU support.")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Fused GRU + SiLU - supports BF16 natively!
        self.gru_silu = GRU_SiLU(
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

    def _gru_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        output, h_final = self.gru_silu(x_proj, state=h0)
        return output, h_final

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: (batch, seq_len, dim) - batch-first input
            prev_hiddens: Previous hidden state (batch, dim_inner) or None
        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, dim_inner) if return_hiddens else None
            next_conv_buffers: None (no conv)
        """
        batch, seq_len, _ = x.shape
        orig_dtype = x.dtype

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Convert to time-first format for haste: (seq_len, batch, dim_inner)
        x_proj_t = x_proj.transpose(0, 1).contiguous()

        # Prepare initial hidden state - haste expects (1, batch, hidden_size)
        if prev_hiddens is not None:
            h0 = prev_hiddens.unsqueeze(0)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=orig_dtype)

        # CHUNKED PROCESSING for gradient checkpointing
        chunk_size = self.recurrence_chunk_size
        gru_outputs = []
        h = h0

        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            x_chunk = x_proj_t[chunk_start:chunk_end]

            if self.use_gradient_checkpointing and self.training:
                chunk_out, h = checkpoint(self._gru_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out, h = self.gru_silu(x_chunk, state=h)

            gru_outputs.append(chunk_out)

        # Concatenate all chunk outputs
        gru_out = torch.cat(gru_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first: (batch, seq_len, dim_inner)
        gru_out = gru_out.transpose(0, 1).contiguous()

        # Normalize after gating
        gru_out = self.post_gate_norm(gru_out)

        # Project output
        output = self.output_proj(gru_out)  # (batch, seq_len, dim)

        if return_hiddens:
            h_final = h.squeeze(0)  # (batch, dim_inner)
            return output, h_final, None
        else:
            return output

    def __repr__(self):
        return f"HasteGRUSiluFused(dim={self.dim}, dim_inner={self.dim_inner}, BF16 native)"

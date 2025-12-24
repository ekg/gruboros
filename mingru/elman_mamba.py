"""ElmanMamba: CLEANER Mamba2-style RNN with haste CUDA kernels.

This is a CLOSER CRIB of Mamba2 than ElmanLeakySelective:
- Input-dependent Δ (timestep) via softplus
- Per-channel decay in log-space: decay_rate = exp(-exp(A_log))
- INPUT-ONLY output gate (NOT h+x!) - like Mamba2's C matrix
- Nonlinear candidate with tanh (our innovation beyond Mamba2)

Key formula:
    candidate = tanh(R @ h + Wx @ x + b)             -- NONLINEAR (our innovation!)
    dt = softplus(W_delta @ x + b_delta)             -- input-dependent timestep
    decay_rate = exp(-exp(A_log))                    -- ALWAYS in (0, 1)!
    alpha = exp(-dt * decay_rate)                    -- per-channel blend factor
    h_new = alpha * h + (1 - alpha) * candidate      -- exponential blend

OUTPUT (configurable via use_output_gate):
    True:  output = h_new * silu(W_gate @ x + b)     -- INPUT-ONLY gate!
    False: output = h_new                            -- no gate, simplest

The key difference from ElmanLeakySelective: NO h+x output gate!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch.elman_variants import ElmanMamba as HasteElmanMamba
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch ElmanMamba not available.")


class RMSNorm(nn.Module):
    """RMSNorm for stabilizing outputs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


class ElmanMamba(nn.Module):
    """
    CLEANER Mamba2-style RNN with haste CUDA kernels.

    Architecture:
        1. CANDIDATE COMPUTATION (NONLINEAR):
           candidate = tanh(R @ h + Wx @ x + b)

        2. MAMBA2-STYLE DISCRETIZATION with log-space A:
           dt = softplus(W_delta @ x + b_delta)    -- input-dependent
           decay_rate = exp(-exp(A_log))           -- ALWAYS in (0, 1)!
           alpha = exp(-dt * decay_rate)           -- per-channel blend
           h_new = alpha * h + (1 - alpha) * candidate

        3. OUTPUT (configurable):
           use_output_gate=True:  output = h_new * silu(W_gate @ x + b)
           use_output_gate=False: output = h_new
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=3.0,  # softplus(3) ≈ 3.0 for meaningful timestep
        A_init_range=(-0.5, 0.5),  # LOG-SPACE: decay_rate=exp(-exp(A))
        use_output_gate=True,  # Whether to use input-only output gate
        **kwargs
    ):
        super().__init__()
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch ElmanMamba is required.")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.delta_init = delta_init
        self.A_init_range = A_init_range
        self.use_output_gate = use_output_gate

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste ElmanMamba (log-space discretization + input-only gate)
        self.elman = HasteElmanMamba(
            self.dim_inner, self.dim_inner,
            delta_init=delta_init,
            A_init_range=A_init_range,
            use_gate=use_output_gate
        )

        # Normalization before output projection
        self.pre_output_norm = RMSNorm(self.dim_inner)

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable residual learning."""
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    def _elman_forward(self, x_proj, h0):
        """Helper for gradient checkpointing."""
        return self.elman(x_proj, h0=h0)

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: (batch, seq_len, dim) - batch-first input
            prev_hiddens: Previous hidden state (batch, dim_inner) or None

        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, dim_inner) if return_hiddens else None
            next_conv_buffers: None
        """
        batch, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Convert to time-first for haste: (seq_len, batch, dim_inner)
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Initial hidden state
        if prev_hiddens is not None:
            h0 = prev_hiddens
        else:
            h0 = torch.zeros(batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # Process in chunks
        chunk_size = self.recurrence_chunk_size
        elman_outputs = []
        h = h0

        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            x_chunk = x_proj[chunk_start:chunk_end]

            if self.use_gradient_checkpointing and self.training:
                chunk_out = checkpoint(self._elman_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out = self.elman(x_chunk, h0=h)

            elman_outputs.append(chunk_out)
            h = chunk_out[-1]

        # Concatenate and convert to batch-first
        elman_out = torch.cat(elman_outputs, dim=0)  # (seq_len, batch, dim_inner)
        elman_out = elman_out.transpose(0, 1).contiguous()  # (batch, seq_len, dim_inner)

        # Normalize before output projection
        elman_out = self.pre_output_norm(elman_out)

        # Project output
        output = self.output_proj(elman_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanMamba(dim={self.dim}, dim_inner={self.dim_inner}, use_output_gate={self.use_output_gate})"

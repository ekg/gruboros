"""ElmanLeakySelective RNN using haste's CUDA-optimized kernels.

Architecture: Nonlinear RNN with Mamba2-style discretization + h+x output gate.

This is our target architecture for achieving Mamba2 parity with nonlinearity.

Key formula:
    candidate = tanh(R @ h + Wx @ x + b)             -- NONLINEAR (our innovation!)
    delta_raw = W_delta @ x + b_delta                -- input-dependent
    dt = softplus(delta_raw)                         -- positive timestep

    # Mamba-style log parameterization for stability:
    decay_rate = exp(-exp(A_log))                    -- ALWAYS in (0, 1)!
    alpha = exp(-dt * decay_rate)                    -- per-channel blend factor

    h_new = alpha * h + (1 - alpha) * candidate      -- exponential blend
    gate = silu(W_gate_x @ x + W_gate_h @ h + b)     -- h+x selective output
    output = h * gate

Key differences from ElmanLeaky:
- Log-space A parameterization (decay_rate = exp(-exp(A_log)) is ALWAYS stable!)
- Per-channel decay rates (like Mamba2's diagonal A matrix)
- Input-dependent timestep via softplus (like Mamba2's Δ)
- Output gate depends on BOTH x AND h (not just x)

The nonlinearity (tanh) in the candidate is our innovation beyond Mamba2's linear dynamics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch.elman_variants import ElmanLeakySelective as HasteElmanLeakySelective
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch not available. ElmanLeakySelective will not work.")


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


class ElmanLeakySelective(nn.Module):
    """
    Nonlinear RNN with Mamba2-style discretization + per-channel decay + h+x output gate.

    Architecture:
        1. CANDIDATE COMPUTATION (NONLINEAR - our innovation!):
           candidate = tanh(R @ h + Wx @ x + b)

        2. MAMBA2-STYLE DISCRETIZATION with log-space A:
           delta_raw = W_delta @ x + b_delta  -- input-dependent (precomputed)
           dt = softplus(delta_raw)            -- positive timestep
           decay_rate = exp(-exp(A_log))       -- ALWAYS in (0, 1) for stability!
           alpha = exp(-dt * decay_rate)       -- per-channel blend factor
           h_new = alpha * h + (1 - alpha) * candidate

        3. h+x OUTPUT SELECTION (computed outside kernel):
           gate = silu(W_gate_x @ x + W_gate_h @ h + b_gate)  -- h AND x!
           output = h * gate

    Key features:
    - Log-space A parameterization for numerical stability
    - Per-channel decay rates (like Mamba2's diagonal A matrix)
    - Input-dependent timestep via softplus (like Mamba2's Δ)
    - NONLINEAR candidate (tanh) - our innovation beyond Mamba2!
    - h+x output gate for richer selectivity

    Uses haste's fused CUDA kernels for fast recurrence.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,  # softplus(-2) ≈ 0.13 for slow dynamics
        A_init_range=(0.5, 2.0),  # Log-space: decay_rate = exp(-exp(A_log)) always in (0,1)
        **kwargs
    ):
        super().__init__()
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch is required for ElmanLeakySelective. Install with: pip install haste_pytorch")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.delta_init = delta_init
        self.A_init_range = A_init_range

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste ElmanLeakySelective with Mamba2-style discretization + h+x output gate
        self.elman = HasteElmanLeakySelective(
            self.dim_inner, self.dim_inner,
            delta_init=delta_init,
            A_init_range=A_init_range
        )

        # Normalization before output projection (stabilizes training)
        self.pre_output_norm = RMSNorm(self.dim_inner)

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable residual learning."""
        # Input projection: std=0.02
        nn.init.normal_(self.input_proj.weight, std=0.02)

        # Output projection: SMALL init for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    def _elman_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        return self.elman(x_proj, h0=h0)

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True, return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: (batch, seq_len, dim) - batch-first input
            prev_hiddens: Previous hidden state (batch, dim_inner) or None
            prev_conv_buffers: Unused (kept for API compatibility)
            return_hiddens: Whether to return hidden states
            actual_length: Unused (kept for API compatibility)
            doc_boundaries: Unused (kept for API compatibility)

        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, dim_inner) if return_hiddens else None
            next_conv_buffers: None (no conv)
        """
        batch, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Convert to time-first format for haste: (seq_len, batch, dim_inner)
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Prepare initial hidden state
        if prev_hiddens is not None:
            h0 = prev_hiddens  # (batch, dim_inner)
        else:
            h0 = torch.zeros(batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # CHUNKED PROCESSING: Process in fixed chunks
        chunk_size = self.recurrence_chunk_size

        elman_outputs = []
        h = h0

        # Process sequence in fixed-size chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Extract chunk (time-first)
            x_chunk = x_proj[chunk_start:chunk_end]  # (chunk_len, batch, dim_inner)

            # Run ElmanLeakySelective on this chunk
            # haste returns output: (chunk_len, batch, dim_inner)
            # Note: output already has h+x gating applied by the kernel
            if self.use_gradient_checkpointing and self.training:
                chunk_out = checkpoint(self._elman_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out = self.elman(x_chunk, h0=h)

            # chunk_out: (chunk_len, batch, dim_inner)
            elman_outputs.append(chunk_out)
            # Get final hidden state (last output is the final state)
            h = chunk_out[-1]  # (batch, dim_inner)

        # Concatenate all chunk outputs (time-first)
        elman_out = torch.cat(elman_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first: (batch, seq_len, dim_inner)
        elman_out = elman_out.transpose(0, 1).contiguous()

        # Normalize before output projection
        elman_out = self.pre_output_norm(elman_out)

        # Project output
        output = self.output_proj(elman_out)  # (batch, seq_len, dim)

        if return_hiddens:
            return output, h, None  # (output, hiddens, conv_buffers)
        else:
            return output

    def __repr__(self):
        return f"ElmanLeakySelective(dim={self.dim}, dim_inner={self.dim_inner}, delta_init={self.delta_init}, A_init_range={self.A_init_range}, using haste CUDA kernels)"

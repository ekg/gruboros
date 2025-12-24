"""ElmanLeakySilu RNN using haste's CUDA-optimized kernels.

Architecture: Same as ElmanLeaky but with silu instead of tanh:
    candidate = silu(R @ h + Wx @ x + b)          -- silu often beats tanh!
    delta = sigmoid(W_delta @ x + b_delta)        -- input-dependent delta [0,1]
    h_new = (1 - delta) * h + delta * candidate   -- leaky integration
    output_gate = silu(W_gate @ x + b_gate)       -- INPUT-ONLY output selection
    output = h_new * output_gate                  -- selective output

silu(x) = x * sigmoid(x) is often better than tanh in modern architectures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch.elman_variants import ElmanLeakySilu as HasteElmanLeakySilu
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch ElmanLeakySilu not available.")


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


class ElmanLeakySilu(nn.Module):
    """
    ElmanLeakySilu: silu activation + leaky integration + input-only output gate.

    Same as ElmanLeaky but with silu instead of tanh.

    Architecture:
        1. CANDIDATE COMPUTATION:
           candidate = silu(R @ h + Wx @ x + b)  -- silu instead of tanh!

        2. INPUT-DEPENDENT DELTA (inside kernel):
           delta = sigmoid(W_delta @ x + b_delta)
           h_new = (1 - delta) * h + delta * candidate

        3. OUTPUT SELECTION (INPUT-ONLY):
           output_gate = silu(W_gate @ x + b_gate)
           output = h * output_gate

    Uses haste's fused CUDA kernels for fast recurrence.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,  # sigmoid(-2) ≈ 0.12 for slow dynamics
        **kwargs
    ):
        super().__init__()
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch ElmanLeakySilu is required.")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.delta_init = delta_init

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste ElmanLeakySilu (silu + leaky integration)
        self.elman = HasteElmanLeakySilu(self.dim_inner, self.dim_inner, delta_init=delta_init)

        # Normalization before gating
        self.pre_gate_norm = RMSNorm(self.dim_inner)

        # silu OUTPUT gate - INPUT ONLY
        self.gate_x = nn.Linear(dim, self.dim_inner, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.dim_inner))

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable residual learning."""
        nn.init.normal_(self.gate_x.weight, std=0.02)
        nn.init.zeros_(self.gate_bias)
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    def _elman_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
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

        # Normalize before gating
        elman_out = self.pre_gate_norm(elman_out)

        # Apply silu OUTPUT gate - INPUT ONLY
        gate_logits = self.gate_x(x) + self.gate_bias
        gate = F.silu(gate_logits)
        gated_out = elman_out * gate

        # Project output
        output = self.output_proj(gated_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanLeakySilu(dim={self.dim}, dim_inner={self.dim_inner}, delta_init={self.delta_init})"

"""SkipElmanSilu: SkipElman RNN with silu output gate.

Architecture:
1. RECURRENCE via SkipElman (has skip connection for gradient flow!):
   z = sigmoid(Wz @ x + Rz @ h + bx_z + bh_z)   # update gate
   a = tanh(Wa @ x + Ra @ h + bx_a + bh_a)      # candidate
   h_new = z * h + (1-z) * a                     # SKIP CONNECTION!

2. OUTPUT SELECTION (silu, like Mult GRU/Mamba2):
   output_gate = silu(W_h @ h + W_x @ x + b)    # input-dependent selection
   output = h * output_gate

This matches the cuDNN GRU + silu architecture but:
- Simpler than GRU (no reset gate)
- Has skip connection (unlike Elman)
- Separate input/hidden biases (like cuDNN)

Haste's fused CUDA kernels make this 2-3x faster than cuDNN GRU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import SkipElman as HasteSkipElman
    SKIPELMAN_AVAILABLE = True
except ImportError:
    SKIPELMAN_AVAILABLE = False
    print("Warning: SkipElman not available in haste_pytorch. SkipElmanSilu will not work.")


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


class SkipElmanSilu(nn.Module):
    """
    SkipElman + silu OUTPUT gate (matches cuDNN GRU + silu architecture).

    Key advantage over ElmanSilu:
        - SkipElman: h_new = z * h_prev + (1-z) * a  - DIRECT h_prev carry!
        - Elman:     h_new = tanh(...) * sigmoid(...) - NO direct carry

    The z*h_prev term allows gradients to flow directly backward,
    enabling much better long-range learning than Elman.

    Simpler than full GRU (no reset gate) but retains the gradient highway.
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
        if not SKIPELMAN_AVAILABLE:
            raise RuntimeError("SkipElman not available in haste_pytorch. Rebuild haste with SkipElman support.")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # SkipElman for recurrence (has skip connection!)
        # Note: SkipElman takes (seq_len, batch, dim) format (time-first)
        self.skip_elman = HasteSkipElman(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            batch_first=False,  # We'll convert to time-first
            return_state_sequence=False,  # Only need final state for stateful recurrence
        )

        # Normalization before output gate
        self.pre_gate_norm = RMSNorm(self.dim_inner)

        # silu OUTPUT gate (like cuDNN GRU + silu / Mamba2)
        # Gate depends on BOTH h and RAW x for true input-dependent selection
        self.gate_h = nn.Linear(self.dim_inner, self.dim_inner, bias=False)
        self.gate_x = nn.Linear(dim, self.dim_inner, bias=False)  # RAW input
        self.gate_bias = nn.Parameter(torch.zeros(self.dim_inner))

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize weights (matches cuDNN GRU mult initialization)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable residual learning."""
        # Gate projections: std=0.02
        nn.init.normal_(self.gate_h.weight, std=0.02)
        nn.init.normal_(self.gate_x.weight, std=0.02)
        nn.init.zeros_(self.gate_bias)

        # Input projection: std=0.02
        nn.init.normal_(self.input_proj.weight, std=0.02)

        # Output projection: small init for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    def _skipelman_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        # x_proj: (seq_len, batch, dim_inner) - time-first for haste
        # h0: (1, batch, dim_inner) - haste expects 3D initial state
        output, h_final = self.skip_elman(x_proj, state=h0)
        return output, h_final

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
        orig_dtype = x.dtype

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Convert to time-first format for haste: (seq_len, batch, dim_inner)
        x_proj_t = x_proj.transpose(0, 1).contiguous()

        # Prepare initial hidden state - haste expects (1, batch, hidden_size)
        if prev_hiddens is not None:
            h0 = prev_hiddens.unsqueeze(0)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # CHUNKED PROCESSING: Process in fixed chunks for gradient checkpointing
        chunk_size = self.recurrence_chunk_size

        skipelman_outputs = []
        h = h0

        # Process sequence in fixed-size chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Extract chunk (time-first)
            x_chunk = x_proj_t[chunk_start:chunk_end]  # (chunk_len, batch, dim_inner)

            # Run SkipElman on this chunk
            if self.use_gradient_checkpointing and self.training:
                chunk_out, h = checkpoint(self._skipelman_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out, h = self.skip_elman(x_chunk, state=h)

            # chunk_out: (chunk_len, batch, dim_inner)
            skipelman_outputs.append(chunk_out)

        # Concatenate all chunk outputs (time-first)
        skipelman_out = torch.cat(skipelman_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first: (batch, seq_len, dim_inner)
        skipelman_out = skipelman_out.transpose(0, 1).contiguous()

        # Normalize before gating
        skipelman_out = self.pre_gate_norm(skipelman_out)

        # Apply silu OUTPUT gate (like cuDNN GRU + silu)
        # Gate depends on BOTH h and RAW x (like cuDNN Mult GRU)
        gate_logits = self.gate_h(skipelman_out) + self.gate_x(x) + self.gate_bias
        gate = F.silu(gate_logits)  # silu for selectivity
        gated_out = skipelman_out * gate  # gated output

        # Project output
        output = self.output_proj(gated_out)  # (batch, seq_len, dim)

        if return_hiddens:
            # Extract final hidden state (squeeze the first dim added for haste)
            h_final = h.squeeze(0)  # (batch, dim_inner)
            return output, h_final, None  # (output, hiddens, conv_buffers)
        else:
            return output

    def __repr__(self):
        return f"SkipElmanSilu(dim={self.dim}, dim_inner={self.dim_inner}, using haste SkipElman + silu gate)"

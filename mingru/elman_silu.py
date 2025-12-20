"""ElmanSilu RNN using haste's CUDA-optimized kernels.

Architecture (matches Mult GRU + silu selectivity):
1. Bounded recurrence via ElmanSigmoid (stable)
2. silu OUTPUT gate for input-dependent selection (like Mamba2/Mult GRU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import ElmanSigmoid as HasteElmanSigmoid
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch not available. ElmanSilu will not work.")


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


class ElmanSilu(nn.Module):
    """
    Haste ElmanSigmoid + silu OUTPUT gate (matches Mult GRU + silu architecture).

    Architecture:
        1. RECURRENCE (bounded, stable):
           h_candidate = tanh(R @ h + Wx @ x + b1)
           recur_gate = sigmoid(R @ h + Wx @ x + b2)  -- BOUNDED [0,1]
           h_new = h_candidate * recur_gate

        2. OUTPUT SELECTION (silu, like Mult GRU/Mamba2):
           output_gate = silu(W_out @ h + b_out)      -- input-dependent selection
           output = h * output_gate

    This matches Mult GRU + silu: bounded recurrence + silu output selection.
    Uses haste's fused CUDA kernels (3x faster than cuDNN GRU).

    Performance (T=512, B=64, D=2048, bf16):
        - Elman + silu output: ~920k tok/s
        - cuDNN GRU: ~280k tok/s
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        gate_bias_init=0.0,  # sigmoid(0)=0.5 for recurrence gate
        **kwargs
    ):
        super().__init__()
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch is required for ElmanSilu. Install with: pip install haste_pytorch")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste ElmanSigmoid for STABLE recurrence (bounded sigmoid gate)
        self.elman = HasteElmanSigmoid(self.dim_inner, self.dim_inner)

        # Initialize recurrence gate bias
        if gate_bias_init != 0.0:
            with torch.no_grad():
                self.elman.bias[self.dim_inner:].fill_(gate_bias_init)

        # silu OUTPUT gate (like Mult GRU + silu)
        # Projects h to gate logits, then silu for input-dependent selection
        self.output_gate = nn.Linear(self.dim_inner, self.dim_inner, bias=True)

        # Output projection: dim_inner -> dim
        # ZERO-INITIALIZED for stable residual connection!
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def _elman_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        # x_proj: (seq_len, batch, dim_inner) - time-first for haste
        # h0: (batch, dim_inner)
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

            # Run ElmanSilu on this chunk
            # haste returns output: (chunk_len, batch, dim_inner)
            if self.use_gradient_checkpointing and self.training:
                chunk_out = checkpoint(self._elman_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out = self.elman(x_chunk, h0=h)

            # chunk_out: (chunk_len, batch, dim_inner) - just the outputs
            elman_outputs.append(chunk_out)
            # Get final hidden state (last output is the final state)
            h = chunk_out[-1]  # (batch, dim_inner)

        # Concatenate all chunk outputs (time-first)
        elman_out = torch.cat(elman_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first: (batch, seq_len, dim_inner)
        elman_out = elman_out.transpose(0, 1).contiguous()

        # Apply silu OUTPUT gate (like Mult GRU + silu)
        # This is input-dependent selection AFTER stable recurrence
        gate_logits = self.output_gate(elman_out)  # [B, T, dim_inner]
        gate = F.silu(gate_logits)  # silu for selectivity
        elman_out = elman_out * gate  # gated output

        # Project output (zero-initialized for stable residual)
        output = self.output_proj(elman_out)  # (batch, seq_len, dim)

        if return_hiddens:
            return output, h, None  # (output, hiddens, conv_buffers)
        else:
            return output

    def __repr__(self):
        return f"ElmanSilu(dim={self.dim}, dim_inner={self.dim_inner}, using haste CUDA kernels)"

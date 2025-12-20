"""Haste GRU + silu output gate.

Architecture (matches cuDNN GRU + silu selectivity):
1. RECURRENCE via haste GRU (has proper skip connection like cuDNN):
   z = sigmoid(Wz @ x + Uz @ h + bz)          # update gate
   r = sigmoid(Wr @ x + Ur @ h + br)          # reset gate
   g = tanh(Wg @ x + Ug @ (r * h) + bg)       # candidate
   h_new = z * h + (1 - z) * g                # KEY: direct h carry via z*h term!

2. OUTPUT SELECTION (silu, like Mult GRU/Mamba2):
   output_gate = silu(W_h @ h + W_x @ x + b)  # input-dependent selection
   output = h * output_gate

This exactly matches CuDNNGRU_Mult architecture but uses haste's faster CUDA kernels.
Haste GRU is typically 2-3x faster than cuDNN GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import GRU as HasteGRU
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch not available. HasteGRUSilu will not work.")


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


class HasteGRUSilu(nn.Module):
    """
    Haste GRU + silu OUTPUT gate (matches cuDNN GRU + silu architecture).

    Key difference from ElmanSilu:
        - Elman: h_new = tanh(...) * sigmoid(...) - NO direct h_prev carry
        - GRU:   h_new = z * h_prev + (1-z) * g   - DIRECT h_prev carry!

    The z*h_prev term in GRU allows gradients to flow directly backward,
    enabling much better long-range learning than Elman.

    Performance:
        - Haste GRU is typically 2-3x faster than cuDNN GRU
        - Has same mathematical properties (skip connection, proper gradient flow)
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
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch is required for HasteGRUSilu. Install with: pip install haste_pytorch")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste GRU for recurrence (has proper skip connection!)
        # Note: Haste GRU takes (seq_len, batch, dim) format (time-first)
        # IMPORTANT: Haste GRU must stay in fp32 - it doesn't support bf16!
        self.gru = HasteGRU(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            batch_first=False,  # We'll convert to time-first
            return_state_sequence=True,  # Get all hidden states, not just final
        )
        # Keep GRU in fp32 - will be enforced in _apply override below
        self._gru_dtype = torch.float32

        # Normalization before output gate
        self.pre_gate_norm = RMSNorm(self.dim_inner)

        # silu OUTPUT gate (like cuDNN GRU + silu / Mamba2)
        # Gate depends on BOTH h and x for true input-dependent selection
        self.gate_h = nn.Linear(self.dim_inner, self.dim_inner, bias=False)
        self.gate_x = nn.Linear(dim, self.dim_inner, bias=False)  # RAW input like cuDNN
        self.gate_bias = nn.Parameter(torch.zeros(self.dim_inner))

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize output projection small for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def _apply(self, fn):
        """Override _apply to keep haste GRU in fp32.

        When model.bfloat16() or model.half() is called, this ensures
        the haste GRU module stays in fp32 since it doesn't support
        other dtypes.
        """
        # Apply fn to all modules except gru
        super()._apply(fn)

        # Force GRU back to fp32 (haste doesn't support bf16/fp16)
        self.gru = self.gru.float()
        return self

    def _gru_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        # x_proj: (seq_len, batch, dim_inner) - time-first for haste
        # h0: (1, batch, dim_inner) - haste expects 3D initial state
        output, h_final = self.gru(x_proj, state=h0)
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

        # Haste GRU doesn't support bf16, so we run in fp32 and convert back
        # (similar to how cuDNN GRU mult handles mixed precision)
        gru_dtype = torch.float32  # haste GRU needs fp32
        x_proj_t = x_proj_t.to(gru_dtype)

        # Prepare initial hidden state - haste GRU expects (1, batch, hidden_size)
        if prev_hiddens is not None:
            h0 = prev_hiddens.unsqueeze(0).to(gru_dtype)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=gru_dtype)

        # CHUNKED PROCESSING: Process in fixed chunks for gradient checkpointing
        chunk_size = self.recurrence_chunk_size

        gru_outputs = []
        h = h0

        # Process sequence in fixed-size chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Extract chunk (time-first)
            x_chunk = x_proj_t[chunk_start:chunk_end]  # (chunk_len, batch, dim_inner)

            # Run GRU on this chunk
            if self.use_gradient_checkpointing and self.training:
                chunk_out, h = checkpoint(self._gru_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out, h = self.gru(x_chunk, state=h)

            # chunk_out: (chunk_len, batch, dim_inner)
            gru_outputs.append(chunk_out)

        # Concatenate all chunk outputs (time-first)
        gru_out = torch.cat(gru_outputs, dim=0)  # (seq_len, batch, dim_inner)

        # Convert back to batch-first and original dtype: (batch, seq_len, dim_inner)
        gru_out = gru_out.transpose(0, 1).contiguous().to(orig_dtype)

        # Normalize before gating
        gru_out = self.pre_gate_norm(gru_out)

        # Apply silu OUTPUT gate (like cuDNN GRU + silu)
        # Gate depends on BOTH h and RAW x (like cuDNN Mult GRU)
        gate_logits = self.gate_h(gru_out) + self.gate_x(x) + self.gate_bias
        gate = F.silu(gate_logits)  # silu for selectivity
        gated_out = gru_out * gate  # gated output

        # Project output
        output = self.output_proj(gated_out)  # (batch, seq_len, dim)

        if return_hiddens:
            # Extract final hidden state (squeeze the first dim added for haste)
            # Convert back to original dtype for TBPTT continuity
            h_final = h.squeeze(0).to(orig_dtype)  # (batch, dim_inner)
            return output, h_final, None  # (output, hiddens, conv_buffers)
        else:
            return output

    def __repr__(self):
        return f"HasteGRUSilu(dim={self.dim}, dim_inner={self.dim_inner}, using haste GRU + silu gate)"

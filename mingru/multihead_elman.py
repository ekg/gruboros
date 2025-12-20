"""MultiHeadElman: Multi-head RNN with per-head recurrence matrices.

Key insight: Mamba2 uses 64 scalar decays (one per head). We use 64×64 matrices
per head - 2048x more expressive recurrence while staying tractable.

Architecture:
1. Split hidden state into nheads independent heads
2. Each head has its own R matrix (headdim × headdim) for recurrence
3. Use softsign activation: x/(1+|x|) - gradient-friendly, non-saturating
4. Input-only output gate (like Mamba2 - no hidden state dependency)
5. Cross-head mixing only in input/output projections

Comparison:
- Mamba2: 64 scalar decays = 64 recurrence params
- MultiHeadElman: 32 heads × 64×64 matrices = 131,072 recurrence params
- Full Elman: 2048×2048 matrix = 4,194,304 recurrence params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """RMSNorm for stabilizing outputs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


def softsign(x):
    """Softsign activation: x / (1 + |x|).

    Like tanh but with softer saturation and non-zero gradients everywhere.
    At x=3: tanh gradient = 0.01, softsign gradient = 0.0625 (6x better!)
    """
    return x / (1 + torch.abs(x))


class MultiHeadElman(nn.Module):
    """
    Multi-head Elman RNN with per-head recurrence matrices.

    Each head has its own 64×64 R matrix, giving 2048x more expressive
    recurrence than Mamba2's scalar decays, while being 32x more tractable
    than a full 2048×2048 matrix.

    Architecture:
        1. INPUT PROJECTION (cross-head mixing):
           x_proj = W_in @ x  (dim -> dim_inner)

        2. SPLIT INTO HEADS:
           x_heads = x_proj.view(batch, seq, nheads, headdim)
           h_heads = h.view(batch, nheads, headdim)

        3. PER-HEAD RECURRENCE (independent, can parallelize):
           For each head i:
             candidate = R[i] @ h[i] + Wx[i] @ x[i] + b[i]
             h_new[i] = softsign(candidate)  # gradient-friendly!

        4. OUTPUT SELECTION (INPUT-ONLY, like Mamba2):
           gate = silu(W_gate @ x + b_gate)  # INPUT ONLY, no h!
           output = concat(h_heads) * gate

        5. OUTPUT PROJECTION (cross-head mixing):
           output = W_out @ output  (dim_inner -> dim)

    Args:
        dim: Model dimension
        nheads: Number of heads (default 32)
        headdim: Dimension per head (default 64)
        expansion_factor: Expansion for dim_inner (default 1.0)
        activation: 'softsign' or 'tanh_residual' (default 'softsign')
        use_gradient_checkpointing: Whether to checkpoint recurrence
        recurrence_chunk_size: Chunk size for processing long sequences
    """

    def __init__(
        self,
        dim,
        nheads=32,
        headdim=64,
        expansion_factor=1.0,
        activation='softsign',
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        gate_bias_init=0.0,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.nheads = nheads
        self.headdim = headdim
        self.dim_inner = nheads * headdim  # Determined by nheads × headdim
        self.activation = activation
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

        # Note: dim_inner is determined by nheads × headdim, not expansion_factor
        # This allows explicit control over the multi-head structure

        # Input projection: dim -> dim_inner (cross-head mixing happens here)
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Per-head recurrence weights
        # R[head]: headdim x headdim - recurrent weights
        # Wx[head]: headdim x headdim - input weights
        # b[head]: headdim - bias
        self.R = nn.Parameter(torch.empty(self.nheads, headdim, headdim))
        self.Wx = nn.Parameter(torch.empty(self.nheads, headdim, headdim))
        self.b = nn.Parameter(torch.zeros(self.nheads, headdim))

        # Normalization before output gate
        self.pre_gate_norm = RMSNorm(self.dim_inner)

        # Input-only output gate (like Mamba2!)
        self.gate_x = nn.Linear(dim, self.dim_inner, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.dim_inner))

        # Output projection: dim_inner -> dim (cross-head mixing)
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Per-head recurrence: small init for stability
        # R should start near identity-ish for gradient flow
        for h in range(self.nheads):
            # Initialize R close to identity with small noise
            nn.init.eye_(self.R[h])
            self.R[h].data *= 0.5  # Scale down
            self.R[h].data += torch.randn_like(self.R[h]) * 0.02

            # Wx: standard small init
            nn.init.normal_(self.Wx[h], std=0.02)

        # Bias: zero init
        nn.init.zeros_(self.b)

        # Input/output projections
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.gate_x.weight, std=0.02)
        nn.init.zeros_(self.gate_bias)

        # Output projection: small for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    def _activation(self, x):
        """Apply activation function."""
        if self.activation == 'softsign':
            return softsign(x)
        elif self.activation == 'tanh_residual':
            return x + torch.tanh(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _recurrence_step(self, x_heads, h_heads):
        """
        One step of multi-head recurrence.

        Args:
            x_heads: (batch, nheads, headdim) - input for this timestep
            h_heads: (batch, nheads, headdim) - previous hidden state

        Returns:
            h_new: (batch, nheads, headdim) - new hidden state
        """
        batch = x_heads.shape[0]

        # Per-head recurrence: h_new[i] = activation(R[i] @ h[i] + Wx[i] @ x[i] + b[i])
        # Use einsum for efficient batched per-head matmul

        # R @ h: (nheads, headdim, headdim) @ (batch, nheads, headdim) -> (batch, nheads, headdim)
        Rh = torch.einsum('nhd,bnh->bnd', self.R, h_heads)

        # Wx @ x: (nheads, headdim, headdim) @ (batch, nheads, headdim) -> (batch, nheads, headdim)
        Wxx = torch.einsum('nhd,bnh->bnd', self.Wx, x_heads)

        # Combine and apply activation
        candidate = Rh + Wxx + self.b  # b broadcasts over batch
        h_new = self._activation(candidate)

        return h_new

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Forward pass with multi-head recurrence.

        Args:
            x: (batch, seq_len, dim) - input
            prev_hiddens: (batch, nheads, headdim) or None - previous hidden state

        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, nheads, headdim) if return_hiddens else None
            next_conv_buffers: None (no conv in this model)
        """
        batch, seq_len, _ = x.shape

        # Project input (cross-head mixing)
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Reshape to heads: (batch, seq_len, nheads, headdim)
        x_heads = x_proj.view(batch, seq_len, self.nheads, self.headdim)

        # Initialize hidden state
        if prev_hiddens is not None:
            h = prev_hiddens  # (batch, nheads, headdim)
        else:
            h = torch.zeros(batch, self.nheads, self.headdim,
                          device=x.device, dtype=x.dtype)

        # Process in chunks
        chunk_size = self.recurrence_chunk_size
        outputs = []

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Process each timestep in chunk
            chunk_outputs = []
            for t in range(chunk_start, chunk_end):
                x_t = x_heads[:, t]  # (batch, nheads, headdim)

                if self.use_gradient_checkpointing and self.training:
                    h = checkpoint(self._recurrence_step, x_t, h, use_reentrant=False)
                else:
                    h = self._recurrence_step(x_t, h)

                chunk_outputs.append(h)

            # Stack chunk outputs: (chunk_len, batch, nheads, headdim)
            chunk_out = torch.stack(chunk_outputs, dim=0)
            outputs.append(chunk_out)

        # Concatenate all chunks: (seq_len, batch, nheads, headdim)
        all_outputs = torch.cat(outputs, dim=0)

        # Reshape to (batch, seq_len, dim_inner)
        all_outputs = all_outputs.permute(1, 0, 2, 3).contiguous()
        all_outputs = all_outputs.view(batch, seq_len, self.dim_inner)

        # Normalize before gating
        all_outputs = self.pre_gate_norm(all_outputs)

        # Apply input-only output gate (like Mamba2!)
        gate_logits = self.gate_x(x) + self.gate_bias
        gate = F.silu(gate_logits)
        gated_out = all_outputs * gate

        # Output projection (cross-head mixing)
        output = self.output_proj(gated_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return (f"MultiHeadElman(dim={self.dim}, nheads={self.nheads}, "
                f"headdim={self.headdim}, activation={self.activation}, "
                f"R_params={self.nheads * self.headdim * self.headdim:,})")

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

This version uses haste CUDA kernels for efficient recurrence computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import haste's MultiHeadElman for efficient CUDA recurrence
from haste_pytorch import MultiHeadElman as HasteMultiHeadElman


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

    Each head has its own headdim×headdim R matrix, giving 2048x more expressive
    recurrence than Mamba2's scalar decays, while being 32x more tractable
    than a full 2048×2048 matrix.

    This version uses haste CUDA kernels for efficient recurrence.

    Architecture:
        1. INPUT PROJECTION (cross-head mixing):
           x_proj = W_in @ x  (dim -> dim_inner)

        2. HASTE MULTI-HEAD RECURRENCE:
           For each head i (efficiently via CUDA):
             h_new[i] = activation(R[i] @ h[i] + Wx[i] @ x[i] + b[i])

        3. OUTPUT SELECTION (INPUT-ONLY, like Mamba2):
           gate = silu(W_gate @ x + b_gate)  # INPUT ONLY, no h!
           output = recurrence_output * gate

        4. OUTPUT PROJECTION (cross-head mixing):
           output = W_out @ output  (dim_inner -> dim)

    Args:
        dim: Model dimension
        nheads: Number of heads (default 32)
        headdim: Dimension per head (default 64)
        expansion_factor: Expansion for dim_inner (ignored, dim_inner = nheads * headdim)
        activation: 'softsign', 'tanh_residual', or 'tanh' (default 'softsign')
    """

    def __init__(
        self,
        dim,
        nheads=32,
        headdim=64,
        expansion_factor=1.0,
        activation='softsign',
        use_gradient_checkpointing=False,  # Ignored - haste handles memory
        recurrence_chunk_size=64,  # Ignored - haste processes full sequence
        gate_bias_init=0.0,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.nheads = nheads
        self.headdim = headdim
        self.dim_inner = nheads * headdim
        self.activation = activation

        # Map activation name to haste activation code
        activation_map = {
            'softsign': 0,
            'tanh_residual': 1,
            'tanh': 2
        }
        self.activation_code = activation_map.get(activation, 0)

        # Input projection: dim -> dim_inner (cross-head mixing)
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Haste multi-head recurrence kernel (handles R, Wx, bias internally)
        self.recurrence = HasteMultiHeadElman(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            nheads=nheads,
            activation=self.activation_code,
            batch_first=False  # We'll transpose for haste
        )

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
        """Initialize weights for stable training.

        Key insight: R controls recurrence dynamics.
        - Too contractive (small eigenvalues): info decays too fast
        - Too expansive (large eigenvalues): exploding gradients
        - Target: spectral radius ~0.9 for good gradient flow
        """
        for h in range(self.nheads):
            # Initialize R with orthogonal + scale to spectral radius ~0.9
            # Orthogonal matrices have eigenvalues on unit circle
            nn.init.orthogonal_(self.recurrence.R[h])
            self.recurrence.R.data[h] *= 0.9

            # Add small identity component for stability
            self.recurrence.R.data[h] += 0.1 * torch.eye(
                self.headdim, device=self.recurrence.R.device, dtype=self.recurrence.R.dtype
            )

            # Wx: Xavier init for balanced gradients
            nn.init.xavier_uniform_(self.recurrence.Wx[h])

        # Bias: zero init
        nn.init.zeros_(self.recurrence.bias)

        # Input/output projections
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.gate_x.weight, std=0.02)
        nn.init.zeros_(self.gate_bias)

        # Output projection: small for residual learning
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))

    # Properties to expose R, Wx, b from the haste module (for compatibility)
    @property
    def R(self):
        return self.recurrence.R

    @property
    def Wx(self):
        return self.recurrence.Wx

    @property
    def b(self):
        return self.recurrence.bias

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Forward pass with multi-head recurrence using haste CUDA kernels.

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

        # Transpose for haste: (batch, seq_len, dim_inner) -> (seq_len, batch, dim_inner)
        x_proj_t = x_proj.transpose(0, 1).contiguous()

        # Run haste multi-head recurrence
        # haste expects (seq_len, batch, dim) and h0 of (batch, nheads, headdim)
        recurrence_out, h_final = self.recurrence(x_proj_t, prev_hiddens)

        # Transpose back: (seq_len, batch, dim_inner) -> (batch, seq_len, dim_inner)
        all_outputs = recurrence_out.transpose(0, 1).contiguous()

        # Normalize before gating
        all_outputs = self.pre_gate_norm(all_outputs)

        # Apply input-only output gate (like Mamba2!)
        gate_logits = self.gate_x(x) + self.gate_bias
        gate = F.silu(gate_logits)
        gated_out = all_outputs * gate

        # Output projection (cross-head mixing)
        output = self.output_proj(gated_out)

        if return_hiddens:
            return output, h_final, None
        else:
            return output

    def __repr__(self):
        return (f"MultiHeadElman(dim={self.dim}, nheads={self.nheads}, "
                f"headdim={self.headdim}, activation={self.activation}, "
                f"R_params={self.nheads * self.headdim * self.headdim:,}, "
                f"backend='haste_cuda')")

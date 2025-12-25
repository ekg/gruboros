"""ElmanMamba2Style - Mamba2-style discretization with competition gate.

Uses Mamba2's linear recurrence formula:
h_new = exp(-softplus(Δ)) * h + (1 - exp(-softplus(Δ))) * Bx

Key differences from standard Elman:
1. No tanh nonlinearity in recurrence
2. No R matrix (diagonal A only, implicit via exp decay)
3. Uses softplus for Δ to ensure positive values
4. exp(-Δ) formulation guarantees stability

Combined with compete × silu output gate.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


class ElmanMamba2Style(nn.Module):
    """Mamba2-style linear recurrence with competition × silu gate."""

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,  # Initial Δ value (will use softplus)
        n_groups=32,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.n_groups = n_groups

        assert self.dim_inner % n_groups == 0
        self.group_size = self.dim_inner // n_groups

        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Mamba2-style: B projection (input to state)
        self.B = nn.Linear(self.dim_inner, self.dim_inner, bias=True)

        # Δ projection (discretization parameter, input-dependent)
        self.W_delta = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        # Initialize so softplus(W_delta @ x + bias) ≈ -delta_init
        # We want exp(-Δ) ≈ sigmoid(delta_init) ≈ 0.88 for delta_init=-2
        # So Δ ≈ 0.13, softplus^{-1}(0.13) ≈ -1.8
        nn.init.constant_(self.W_delta.bias, -1.8)

        # NO R matrix! This is the key Mamba2 simplification

        # Gate projections
        self.compete_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.silu_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.gate_scale = nn.Parameter(torch.ones(n_groups) * self.group_size)

        self.pre_out_norm = RMSNorm(self.dim_inner)
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.compete_proj.weight, std=0.02)
        nn.init.normal_(self.silu_proj.weight, std=0.02)
        nn.init.normal_(self.B.weight, std=0.02)
        nn.init.normal_(self.W_delta.weight, std=0.02)

    def _group_softmax(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_groups, self.group_size)
        x = F.softmax(x, dim=-1)
        x = x * self.gate_scale.view(1, 1, self.n_groups, 1)
        return x.view(B, T, D)

    def _mamba2_step(self, x_t, h):
        """Single Mamba2-style step: linear recurrence, no tanh, no R."""
        # Discretization parameter
        delta = F.softplus(self.W_delta(x_t))  # Always positive

        # Mamba2 formula: h = exp(-Δ) * h + (1 - exp(-Δ)) * Bx
        # Note: no tanh, no R @ h!
        decay = torch.exp(-delta)
        Bx = self.B(x_t)

        h_new = decay * h + (1 - decay) * Bx
        return h_new

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        batch, seq_len, _ = x.shape

        # Hybrid gate
        compete_gate = self._group_softmax(self.compete_proj(x))
        silu_gate = F.silu(self.silu_proj(x))
        gate = compete_gate * silu_gate

        x_proj = self.input_proj(x)

        if prev_hiddens is not None:
            h = prev_hiddens
        else:
            h = torch.zeros(batch, self.dim_inner, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            h = self._mamba2_step(x_proj[:, t], h)
            outputs.append(h)

        elman_out = torch.stack(outputs, dim=1)

        elman_out = elman_out * gate
        elman_out = self.pre_out_norm(elman_out)
        output = self.output_proj(elman_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanMamba2Style(dim={self.dim}, n_groups={self.n_groups})"

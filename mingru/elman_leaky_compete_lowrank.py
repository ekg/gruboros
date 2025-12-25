"""ElmanLeakyCompeteLowRank - Competition gate with low-rank R matrix.

R = U @ V where U is dim×rank and V is rank×dim.
This saves params and can be reinvested in more depth.

For dim=2048, rank=512:
- Full R: 2048×2048 = 4.2M params
- Low-rank R: 2048×512 + 512×2048 = 2.1M params (50% savings!)
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


class ElmanLeakyCompeteLowRank(nn.Module):
    """Competition × silu with low-rank R = U @ V."""

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,
        n_groups=32,
        r_rank=512,  # Rank of R matrix factorization
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.n_groups = n_groups
        self.r_rank = r_rank

        assert self.dim_inner % n_groups == 0
        self.group_size = self.dim_inner // n_groups

        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Elman recurrence with LOW-RANK R
        self.W_x = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        # R = U @ V (low-rank factorization)
        self.R_U = nn.Linear(self.dim_inner, r_rank, bias=False)
        self.R_V = nn.Linear(r_rank, self.dim_inner, bias=False)

        # Delta gate
        self.W_delta = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        nn.init.constant_(self.W_delta.bias, delta_init)

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
        nn.init.normal_(self.W_x.weight, std=0.02)
        # Initialize low-rank factors
        nn.init.normal_(self.R_U.weight, std=0.02)
        nn.init.normal_(self.R_V.weight, std=0.02)
        nn.init.normal_(self.W_delta.weight, std=0.02)

    def _group_softmax(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_groups, self.group_size)
        x = F.softmax(x, dim=-1)
        x = x * self.gate_scale.view(1, 1, self.n_groups, 1)
        return x.view(B, T, D)

    def _elman_step(self, x_t, h):
        """Single Elman step with low-rank R."""
        # Low-rank recurrence: R @ h = V @ (U @ h)
        Rh = self.R_V(self.R_U(h))
        candidate = torch.tanh(self.W_x(x_t) + Rh)

        delta = torch.sigmoid(self.W_delta(x_t))
        h_new = (1 - delta) * h + delta * candidate
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
            h = self._elman_step(x_proj[:, t], h)
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
        return f"ElmanLeakyCompeteLowRank(dim={self.dim}, rank={self.r_rank})"

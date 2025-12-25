"""ElmanLeakyCompeteLearnedDelta - Competition gate with learned delta scaling.

Adds a learned per-dimension scaling factor to the delta (decay rate).
effective_delta = base_delta * learned_scale

This allows the model to learn different timescales per dimension.
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


class ElmanLeakyCompeteLearnedDelta(nn.Module):
    """Competition × silu with learned per-dim delta scaling.

    Instead of using Haste's fixed delta_init, we implement the Elman
    recurrence directly with a learned delta scale per dimension.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,
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

        # Elman recurrence parameters (normally in Haste, now explicit)
        self.W_x = nn.Linear(self.dim_inner, self.dim_inner, bias=True)  # Input to candidate
        self.R = nn.Linear(self.dim_inner, self.dim_inner, bias=False)   # Recurrent weight

        # Delta gate parameters
        self.W_delta = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        # Initialize delta bias so sigmoid(bias) ≈ 0.12
        nn.init.constant_(self.W_delta.bias, delta_init)

        # LEARNED per-dimension delta scale (log-space for positivity)
        # Initialized to 0 so exp(0) = 1 (no scaling initially)
        self.log_delta_scale = nn.Parameter(torch.zeros(self.dim_inner))

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
        nn.init.orthogonal_(self.R.weight)  # Orthogonal init for recurrent
        nn.init.normal_(self.W_delta.weight, std=0.02)

    def _group_softmax(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_groups, self.group_size)
        x = F.softmax(x, dim=-1)
        x = x * self.gate_scale.view(1, 1, self.n_groups, 1)
        return x.view(B, T, D)

    def _elman_step(self, x_t, h):
        """Single Elman step with learned delta scaling."""
        # Candidate: tanh(Wx @ x + R @ h)
        candidate = torch.tanh(self.W_x(x_t) + self.R(h))

        # Delta with learned scaling
        base_delta = torch.sigmoid(self.W_delta(x_t))
        delta_scale = torch.exp(self.log_delta_scale)  # Per-dim scale
        delta = base_delta * delta_scale
        delta = delta.clamp(0, 1)  # Ensure valid range

        # Leaky integration
        h_new = (1 - delta) * h + delta * candidate
        return h_new

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        batch, seq_len, _ = x.shape

        # Hybrid gate: competition × silu
        compete_gate = self._group_softmax(self.compete_proj(x))
        silu_gate = F.silu(self.silu_proj(x))
        gate = compete_gate * silu_gate

        x_proj = self.input_proj(x)

        if prev_hiddens is not None:
            h = prev_hiddens
        else:
            h = torch.zeros(batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # Process sequence
        outputs = []
        for t in range(seq_len):
            h = self._elman_step(x_proj[:, t], h)
            outputs.append(h)

        elman_out = torch.stack(outputs, dim=1)  # [B, T, D]

        elman_out = elman_out * gate
        elman_out = self.pre_out_norm(elman_out)
        output = self.output_proj(elman_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanLeakyCompeteLearnedDelta(dim={self.dim}, n_groups={self.n_groups})"

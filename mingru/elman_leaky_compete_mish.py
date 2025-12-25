"""ElmanLeakyCompeteMish - Competition gate × mish hybrid.

Combines group softmax competition with element-wise mish gating.
gate = group_softmax(W1 @ x) * mish(W2 @ x)

Mish = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
Mish is self-regularizing and has smoother gradients than ReLU/silu.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch.elman_variants import ElmanLeaky as HasteElmanLeaky
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


def mish(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * torch.tanh(F.softplus(x))


class ElmanLeakyCompeteMish(nn.Module):
    """Competition × mish hybrid gate."""

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
        if not HASTE_AVAILABLE:
            raise RuntimeError("haste_pytorch required")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size
        self.delta_init = delta_init
        self.n_groups = n_groups

        assert self.dim_inner % n_groups == 0
        self.group_size = self.dim_inner // n_groups

        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.elman = HasteElmanLeaky(self.dim_inner, self.dim_inner, delta_init=delta_init)

        # Two separate gate projections
        self.compete_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.mish_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Scale per group
        self.gate_scale = nn.Parameter(torch.ones(n_groups) * self.group_size)

        self.pre_out_norm = RMSNorm(self.dim_inner)
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.compete_proj.weight, std=0.02)
        nn.init.normal_(self.mish_proj.weight, std=0.02)

    def _elman_forward(self, x_proj, h0):
        return self.elman(x_proj, h0=h0)

    def _group_softmax(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_groups, self.group_size)
        x = F.softmax(x, dim=-1)
        x = x * self.gate_scale.view(1, 1, self.n_groups, 1)
        return x.view(B, T, D)

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        batch, seq_len, _ = x.shape

        # Hybrid gate: competition × mish
        compete_gate = self._group_softmax(self.compete_proj(x))
        mish_gate = mish(self.mish_proj(x))
        gate = compete_gate * mish_gate

        x_proj = self.input_proj(x)
        x_proj = x_proj.transpose(0, 1).contiguous()

        if prev_hiddens is not None:
            h0 = prev_hiddens
        else:
            h0 = torch.zeros(batch, self.dim_inner, device=x.device, dtype=x.dtype)

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

        elman_out = torch.cat(elman_outputs, dim=0)
        elman_out = elman_out.transpose(0, 1).contiguous()

        elman_out = elman_out * gate
        elman_out = self.pre_out_norm(elman_out)
        output = self.output_proj(elman_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanLeakyCompeteMish(dim={self.dim}, n_groups={self.n_groups})"

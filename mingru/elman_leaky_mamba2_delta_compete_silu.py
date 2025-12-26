"""ElmanLeakyMamba2DeltaCompeteSilu - Mamba2-style delta + compete×silu gate.

Combines:
1. Haste ElmanLeakyMamba2Delta kernel (softplus/exp delta parameterization)
2. Compete×silu output gate (like our baseline)

Architecture:
    # Core recurrence (Haste CUDA kernel):
    candidate = tanh(R @ h + Wx @ x + b)
    delta = softplus(W_delta @ x + b_delta)    -- log-space parameterization
    decay = exp(-delta)                         -- between 0 and 1
    h_new = decay * h + (1 - decay) * candidate

    # Output gate (Python):
    gate = group_softmax(W1 @ x) * silu(W2 @ x)
    output = W_out @ norm(h_new * gate)

Key differences from ElmanLeakyCompeteSilu (baseline):
- Uses softplus/exp instead of sigmoid for delta
- Wider dynamic range, better gradients
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from haste_pytorch import ElmanLeakyMamba2Delta as HasteElmanLeakyMamba2Delta
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


class ElmanLeakyMamba2DeltaCompeteSilu(nn.Module):
    """Mamba2-style delta + compete×silu gate.

    Uses Haste CUDA kernel for fast recurrence with log-space delta.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-1.8,  # softplus(-1.8) ≈ 0.15, exp(-0.15) ≈ 0.86 decay
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
        self.elman = HasteElmanLeakyMamba2Delta(
            self.dim_inner, self.dim_inner, delta_init=delta_init
        )

        # Two separate gate projections (compete × silu)
        self.compete_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.silu_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Scale per group
        self.gate_scale = nn.Parameter(torch.ones(n_groups) * self.group_size)

        self.pre_out_norm = RMSNorm(self.dim_inner)
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.compete_proj.weight, std=0.02)
        nn.init.normal_(self.silu_proj.weight, std=0.02)

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

        # Hybrid gate: competition × silu
        compete_gate = self._group_softmax(self.compete_proj(x))
        silu_gate = F.silu(self.silu_proj(x))
        gate = compete_gate * silu_gate

        x_proj = self.input_proj(x)
        x_proj = x_proj.transpose(0, 1).contiguous()  # [T, B, D]

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
        elman_out = elman_out.transpose(0, 1).contiguous()  # [B, T, D]

        elman_out = elman_out * gate
        elman_out = self.pre_out_norm(elman_out)
        output = self.output_proj(elman_out)

        if return_hiddens:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"ElmanLeakyMamba2DeltaCompeteSilu(dim={self.dim}, n_groups={self.n_groups})"

"""ElmanLeakyCompeteAsymmetric - Asymmetric competition gate.

Competition operates on fewer groups (coarser), silu stays full dimension.
gate = group_softmax(W1 @ x, n_groups=16) * silu(W2 @ x)

The idea: competition benefits from coarser grouping (more dims competing),
while silu benefits from full per-dimension control.
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


class ElmanLeakyCompeteAsymmetric(nn.Module):
    """Asymmetric gate: coarse competition × full-dim silu."""

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        recurrence_chunk_size=64,
        delta_init=-2.0,
        n_groups=16,  # Coarser competition (128 dims per group for dim=2048)
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

        # Competition projection - outputs n_groups values, broadcast to full dim
        self.compete_proj = nn.Linear(dim, n_groups, bias=False)
        # Silu projection - full dimension
        self.silu_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Scale per group (for competition output)
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

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        batch, seq_len, _ = x.shape

        # Asymmetric gate: coarse competition × full silu
        # Competition: [B, T, n_groups] -> softmax -> scale -> broadcast to [B, T, dim_inner]
        compete_logits = self.compete_proj(x)  # [B, T, n_groups]
        compete_probs = F.softmax(compete_logits, dim=-1)  # softmax over groups
        compete_probs = compete_probs * self.gate_scale.view(1, 1, self.n_groups)
        # Broadcast each group value to all dims in that group
        compete_gate = compete_probs.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
        compete_gate = compete_gate.reshape(batch, seq_len, self.dim_inner)

        # Silu: full dimension
        silu_gate = F.silu(self.silu_proj(x))

        gate = compete_gate * silu_gate

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
        return f"ElmanLeakyCompeteAsymmetric(dim={self.dim}, n_groups={self.n_groups})"

"""
ElmanTripleRCompeteSilu: Triple R matrices + compete×silu output gate

Full architecture matching baseline (with input/output projections):
  # Input projection
  x_proj = input_proj(x)

  # Recurrence (using Haste kernel)
  candidate = tanh(R_h @ h + R_x @ x_proj + b)
  delta = sigmoid(R_delta @ h + W_delta @ x_proj + b_delta)
  h_new = (1 - delta) * h + delta * candidate

  # Output gate (outside kernel)
  gate = group_softmax(compete_proj(x)) * silu(silu_proj(x))
  output = output_proj(norm(h_new * gate))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mingru.elman_triple_r import ElmanTripleR


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.g


class ElmanTripleRCompeteSilu(nn.Module):
    """
    ElmanTripleR with compete×silu output gate.
    Matches baseline architecture with input/output projections.

    Args:
        dim: Size of input/output features
        expansion_factor: Internal dimension multiplier (default 1.0)
        n_groups: Number of groups for softmax competition
        delta_init: Initial value for delta bias
    """

    def __init__(self, dim, expansion_factor=1.0, n_groups=32, delta_init=-2.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.hidden_size = dim  # For compatibility
        self.n_groups = n_groups

        assert self.dim_inner % n_groups == 0
        self.group_size = self.dim_inner // n_groups

        # Input projection
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Core RNN with Triple R matrices
        self.rnn = ElmanTripleR(self.dim_inner, self.dim_inner, delta_init=delta_init)

        # Gate projections (from original dim, like baseline)
        self.compete_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.silu_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Scale per group (like baseline)
        self.gate_scale = nn.Parameter(torch.ones(n_groups) * self.group_size)

        # Output
        self.pre_out_norm = RMSNorm(self.dim_inner)
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.compete_proj.weight)
        nn.init.xavier_uniform_(self.silu_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: Input tensor [B, T, dim] (batch first for compatibility)
            prev_hiddens: Previous hidden state [B, dim_inner] or None
            prev_conv_buffers: Unused (for API compatibility)
            return_hiddens: If True, return hidden states
            return_next_prev_hidden: If True, return next hidden state
            actual_length: Unused (for API compatibility)
            doc_boundaries: Unused (for API compatibility)

        Returns:
            output: Gated output [B, T, dim]
            h_final: Final hidden state [B, dim_inner]
        """
        B, T, _ = x.shape

        # Input projection
        x_proj = self.input_proj(x)  # [B, T, dim_inner]

        # Convert to [T, B, dim_inner] for RNN
        x_rnn = x_proj.transpose(0, 1).contiguous()

        h0 = prev_hiddens if prev_hiddens is not None else None

        # Run RNN
        h = self.rnn(x_rnn, h0)  # [T+1, B, dim_inner]

        # Compute output gate (from original x, like baseline)
        # Group softmax competition
        compete_raw = self.compete_proj(x)  # [B, T, dim_inner]
        compete_raw = compete_raw.view(B, T, self.n_groups, -1)  # [B, T, n_groups, group_size]
        compete = F.softmax(compete_raw, dim=-1)  # Softmax within groups
        compete = compete.view(B, T, self.dim_inner)

        # SiLU activation
        silu_out = F.silu(self.silu_proj(x))  # [B, T, dim_inner]

        # Combined gate with scale
        gate_scale = self.gate_scale.view(1, 1, self.n_groups, 1).expand(B, T, -1, self.group_size)
        gate_scale = gate_scale.reshape(B, T, self.dim_inner)
        gate = compete * silu_out * gate_scale  # [B, T, dim_inner]

        # Apply gate to hidden states (skip h[0] which is initial state)
        h_out = h[1:].transpose(0, 1)  # [B, T, dim_inner]
        gated = h_out * gate

        # Normalize and project output
        output = self.output_proj(self.pre_out_norm(gated))  # [B, T, dim]

        h_final = h[-1]  # [B, dim_inner]

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, dim_inner={self.dim_inner}, n_groups={self.n_groups}'

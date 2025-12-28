"""
Level 6 FAST: Triple R Elman (no log-space overhead)

Same architecture as logspace_triple_r but WITHOUT log-space computation.
Uses cuBLAS gemm for all R @ h operations.

This is the most expressive variant with THREE R matrices:
- R_h: Hidden state recurrence
- R_x: Replaces W_x for input transformation
- R_delta: Delta gate modulation from hidden state

Recurrence:
    v_t = R_x @ x + R_h @ h_{t-1} + b
    delta_raw = W_delta @ x + R_delta @ h_{t-1} + b_delta
    delta_t = sigmoid(delta_raw)
    h_t = (1 - delta_t) * h_{t-1} + delta_t * tanh(v_t)
    output_t = compete(h_t) * silu(W_out @ h_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVEL_6_FAST_AVAILABLE = True


class TripleRElmanCell(nn.Module):
    """
    Triple R Elman cell - Level 6 FAST.

    Three R matrices for maximum expressivity:
    - R_h: h_{t-1} contribution to candidate
    - R_x: Input transformation (replaces W_x)
    - R_delta: h_{t-1} contribution to delta gate

    No log-space overhead - pure linear computation.
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        assert dim % n_groups == 0

        # Candidate computation: v = R_x @ x + R_h @ h + b
        self.R_x = nn.Parameter(torch.empty(dim, dim))
        self.R_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta gate: delta = sigmoid(W_delta @ x + R_delta @ h + b_delta)
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.R_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier init for all matrices
        nn.init.xavier_uniform_(self.R_x)
        nn.init.xavier_uniform_(self.W_delta)
        nn.init.xavier_uniform_(self.W_out)

        # Initialize R_h and R_delta close to identity for stability
        with torch.no_grad():
            nn.init.eye_(self.R_h)
            self.R_h.mul_(0.9)

            nn.init.eye_(self.R_delta)
            self.R_delta.mul_(0.1)  # Small contribution from h to delta

    def forward(self, x, h0=None):
        """
        Forward pass.

        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            h: [T+1, B, dim] hidden states
            output: [T, B, dim] outputs
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Candidate: v = R_x @ x + R_h @ h + b
            Rx_x = x_t @ self.R_x.T
            Rh_h = h_prev @ self.R_h.T
            v = Rx_x + Rh_h + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + R_delta @ h + b_delta)
            Wdelta_x = x_t @ self.W_delta.T
            Rdelta_h = h_prev @ self.R_delta.T
            delta_raw = Wdelta_x + Rdelta_h + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update
            h_new = (1 - delta) * h_prev + delta * candidate
            h_list.append(h_new)

            # Compete × silu output
            h_grouped = h_new.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_grouped, dim=-1)
            compete = compete.view(B, D)

            out_proj = h_new @ self.W_out.T
            output = compete * F.silu(out_proj)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)

        return h, output


class TripleRElman(nn.Module):
    """
    Triple R Elman layer - Level 6 FAST with projections.
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Triple R cell
        self.cell = TripleRElmanCell(
            dim=self.d_inner,
            n_groups=n_groups,
            delta_init=delta_init,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, prev_hidden=None):
        """
        Args:
            x: [B, T, dim] input
            prev_hidden: [B, d_inner] previous hidden state

        Returns:
            output: [B, T, dim]
            final_hidden: [B, d_inner]
        """
        B, T, D = x.shape

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose for cell: [T, B, d_inner]
        x_proj = x_proj.transpose(0, 1)

        # Run cell
        h, cell_output = self.cell(x_proj, prev_hidden)

        # Transpose back and project: [B, T, d_inner] -> [B, T, dim]
        cell_output = cell_output.transpose(0, 1)
        output = self.out_proj(cell_output)
        output = self.dropout(output)

        # Final hidden is last hidden state
        final_hidden = h[-1]  # [B, d_inner]

        return output, final_hidden

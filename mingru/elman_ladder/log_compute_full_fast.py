"""
Level 5 FAST: Full R Matrix Elman (no log-space overhead)

Same architecture as log_compute_full but WITHOUT log-space computation.
Uses cuBLAS gemm for R @ h, stores everything in linear.

This is essentially Level 3 (Diagonal Selective) but with FULL R matrix
instead of diagonal r_h.

Recurrence:
    delta = sigmoid(W_delta @ x_t + b_delta)
    candidate = tanh(W_x @ x_t + R @ h_{t-1} + b)  # Full R matrix!
    h_t = (1 - delta) * h_{t-1} + delta * candidate
    output_t = compete(h_t) * silu(W_out @ h_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVEL_5_FAST_AVAILABLE = True


class FullRElmanCell(nn.Module):
    """
    Full R Matrix Elman cell - Level 5 FAST.

    Like Level 3 but with full R matrix instead of diagonal.
    No log-space overhead - pure linear computation.
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        assert dim % n_groups == 0

        # Candidate computation
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.R = nn.Parameter(torch.empty(dim, dim))  # FULL R matrix
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier init for W matrices
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_delta)
        nn.init.xavier_uniform_(self.W_out)

        # Initialize R close to identity for stability
        with torch.no_grad():
            nn.init.eye_(self.R)
            self.R.mul_(0.9)  # Slight decay

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

            # Delta gate
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Candidate with FULL R matrix (cuBLAS gemm under the hood)
            Rh = h_prev @ self.R.T  # Full matrix multiply
            candidate_raw = x_t @ self.W_x.T + Rh + self.b
            candidate = torch.tanh(candidate_raw)

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


class FullRElman(nn.Module):
    """
    Full R Matrix Elman layer - Level 5 FAST with projections.
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Full R cell
        self.cell = FullRElmanCell(
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

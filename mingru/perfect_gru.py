"""
Perfect GRU - EXACT cuDNN math with memory efficiency

Approach: Pure PyTorch but with per-timestep operations
Goal: 100% mathematical equivalence to cuDNN GRU
"""

import torch
import torch.nn as nn


class PerfectGRU(nn.Module):
    """
    GRU with EXACT cuDNN mathematical equivalence.

    Key: Match PyTorch nn.GRU equations EXACTLY:
    r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    """

    def __init__(self, dim, expansion_factor=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Match StandardGRU EXACTLY
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)
        self.input_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner, bias=False)
        self.U_recurrent = nn.Linear(self.dim_inner, 3 * self.dim_inner, bias=False)

        self.bias_ih = nn.Parameter(torch.zeros(3 * self.dim_inner))
        self.bias_hh = nn.Parameter(torch.zeros(3 * self.dim_inner))

        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

        print(f"[PerfectGRU] dim={dim}, dim_inner={self.dim_inner}")
        print(f"[PerfectGRU] EXACT cuDNN math - testing for perfect equivalence")

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True,
                return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Forward with EXACT cuDNN GRU equations.

        This implementation prioritizes mathematical correctness over speed.
        Once verified correct, we can optimize.
        """
        B, T, D = x.shape
        H = self.dim_inner
        device = x.device
        dtype = x.dtype

        # Project input
        x_proj = self.input_proj(x)  # [B, T, H]
        input_gates = self.input_projection(x_proj)  # [B, T, 3*H]

        # Initialize hidden state
        if prev_hiddens is not None:
            h = prev_hiddens
        else:
            h = torch.zeros(B, H, device=device, dtype=dtype)

        # Preallocate output
        h_all = torch.empty(B, T, H, device=device, dtype=dtype)

        # Sequential loop - EXACT cuDNN math
        for t in range(T):
            # Handle document boundaries
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t].unsqueeze(1)  # [B, 1]
                h = h * (~reset_mask).float()

            # Get input contribution for this timestep [B, 3*H]
            x_t = input_gates[:, t, :]

            # Compute recurrent contribution [B, 3*H]
            h_rec = self.U_recurrent(h)

            # Split gates - MUST match PyTorch GRU order: reset, update, new
            # PyTorch GRU uses [rzn] order
            i_r, i_z, i_n = x_t.chunk(3, dim=1)
            r_rec, z_rec, n_rec = h_rec.chunk(3, dim=1)
            b_ih_r, b_ih_z, b_ih_n = self.bias_ih.chunk(3)
            b_hh_r, b_hh_z, b_hh_n = self.bias_hh.chunk(3)

            # EXACT PyTorch GRU equations:
            # r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
            r = torch.sigmoid(i_r + r_rec + b_ih_r + b_hh_r)

            # z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
            z = torch.sigmoid(i_z + z_rec + b_ih_z + b_hh_z)

            # n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))
            # KEY: Reset gate multiplies (W_hn @ h + b_hn), NOT just W_hn @ h!
            n = torch.tanh(i_n + b_ih_n + r * (n_rec + b_hh_n))

            # h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
            h = (1 - z) * n + z * h

            # Save to output
            h_all[:, t, :] = h

        # Project output
        output = self.output_proj(h_all)

        # Match StandardGRU return signature
        if return_hiddens or return_next_prev_hidden:
            return output, h, None
        else:
            return output

    def __repr__(self):
        return f"PerfectGRU(dim={self.dim}, EXACT cuDNN math)"

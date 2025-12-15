"""
Selective GRU - Input-Dependent Gating

Inspired by Mamba's selective SSM, adds input-dependent selection mechanisms:
1. Δ(x): Learned discretization step per timestep (how much to update)
2. s(x): Selection vector that modulates which dimensions to update

Standard GRU:
  r = σ(W_r x + U_r h)
  z = σ(W_z x + U_z h)
  n = tanh(W_n x + r ⊙ U_n h)
  h' = (1-z) ⊙ n + z ⊙ h

Selective GRU:
  Δ = softplus(W_Δ x)  # Input-dependent update magnitude
  s = σ(W_s x)         # Input-dependent selection (which dims to update)

  r = σ(W_r x + U_r h)
  z = σ(W_z x + U_z h + Δ)  # Δ modulates forget gate
  n = tanh(W_n x + r ⊙ U_n h)
  h' = s ⊙ [(1-z) ⊙ n + z ⊙ h] + (1-s) ⊙ h  # Selective update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveGRU(nn.Module):
    """
    GRU with selective input-dependent mechanisms.

    Uses cuDNN GRU as base, but adds:
    - Δ(x): Learned timestep discretization
    - s(x): Selection vector for dimension-wise updates
    """

    def __init__(self, dim, expansion_factor=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Input projection: dim -> 3*dim_inner (for r, z, n gates)
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=False)

        # Selection mechanisms
        # Δ(x): discretization step (how much to update) - projects to dim_inner
        self.delta_projection = nn.Linear(dim, self.dim_inner, bias=True)

        # s(x): selection vector (which dimensions to update) - projects to dim_inner
        self.selection_projection = nn.Linear(dim, self.dim_inner, bias=True)

        # GRU cell (using PyTorch's optimized cuDNN implementation)
        self.gru = nn.GRU(
            input_size=3 * self.dim_inner,  # Preprocessed gates
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bias=True
        )

        # Output projection
        self.output_projection = nn.Linear(self.dim_inner, dim, bias=False)

        print(f"[SelectiveGRU] dim={dim}, dim_inner={self.dim_inner}")
        print(f"[SelectiveGRU] Adds input-dependent Δ(x) and s(x) selection")

    def forward(self, x, prev_hiddens=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Args:
            x: [B, T, D]
            prev_hiddens: [B, H]
            doc_boundaries: [B, T] bool

        Returns:
            output: [B, T, D]
            h_final: [B, H] (if return_next_prev_hidden)
        """
        B, T, D = x.shape
        H = self.dim_inner
        device = x.device
        dtype = x.dtype

        # Compute selective mechanisms
        # Δ(x): timestep/update magnitude [B, T, H]
        # Use softplus to ensure Δ > 0
        delta = F.softplus(self.delta_projection(x))  # [B, T, H]

        # s(x): selection vector [B, T, H]
        # Sigmoid to get [0, 1] range (0 = skip update, 1 = full update)
        selection = torch.sigmoid(self.selection_projection(x))  # [B, T, H]

        # Precompute input gates [B, T, 3*H]
        input_gates = self.input_projection(x)

        # Add delta to z gate (modulate forget gate based on input)
        # input_gates: [r | z | n] each of size H
        # Add delta to middle third (z gate)
        input_gates[:, :, H:2*H] = input_gates[:, :, H:2*H] + delta

        # Initialize hidden state
        if prev_hiddens is None:
            h0 = torch.zeros(1, B, H, device=device, dtype=dtype)
        else:
            h0 = prev_hiddens.unsqueeze(0)  # [1, B, H]

        # Handle document boundaries
        if doc_boundaries is not None:
            # Process sequence with resets
            outputs = []
            h_curr = h0

            for t in range(T):
                # Check if we need to reset
                reset_mask = doc_boundaries[:, t].unsqueeze(0).unsqueeze(2)  # [1, B, 1]
                h_curr = h_curr * (1.0 - reset_mask.to(dtype))

                # Process single timestep
                x_t = input_gates[:, t:t+1, :]  # [B, 1, 3*H]
                h_out, h_curr = self.gru(x_t, h_curr)  # h_out: [B, 1, H]

                # Apply selection: h' = s ⊙ h_new + (1-s) ⊙ h_old
                s_t = selection[:, t:t+1, :]  # [B, 1, H]
                h_prev = h0.squeeze(0) if t == 0 else outputs[-1]
                h_selected = s_t * h_out + (1.0 - s_t) * h_prev.unsqueeze(1)

                outputs.append(h_selected)

            h_all = torch.cat(outputs, dim=1)  # [B, T, H]
            h_final = h_curr

        else:
            # Standard forward pass (no document boundaries)
            h_all, h_final = self.gru(input_gates, h0)  # h_all: [B, T, H]

            # Apply selective update: h' = s ⊙ h_new + (1-s) ⊙ h_old
            # For selective update, we need previous hidden states
            # Shift h_all by one timestep to get h_old
            h_prev = torch.cat([h0.squeeze(0).unsqueeze(1), h_all[:, :-1, :]], dim=1)  # [B, T, H]
            h_all = selection * h_all + (1.0 - selection) * h_prev

        # Project output
        output = self.output_projection(h_all)

        if return_next_prev_hidden:
            return output, h_final.squeeze(0)
        else:
            return output

    def __repr__(self):
        return f"SelectiveGRU(dim={self.dim}, dim_inner={self.dim_inner}, selective=True)"

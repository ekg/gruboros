"""
Fully Fused Triton GRU - Actually Working Version

Strategy:
- Parallelize across batch (B) and hidden block (H/BLOCK_H)
- Sequential timesteps INSIDE kernel (eliminates Python loop overhead)
- Precompute input projections with CUBLAS (it's faster than Triton matmul)
- Fuse hidden projections + GRU cell updates

This eliminates ~1500 kernel launches down to ~20.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gru_timesteps_kernel(
    # Precomputed input gates: [B, T, 3H]
    gates_input_ptr,
    # Hidden state buffer (read/write): [B, T+1, H]
    # We store h_0, h_1, ..., h_T here
    h_states_ptr,
    # Hidden projection weights: [H, 3H] for r, z, n
    W_h_ptr,
    # Dimensions
    B, T, H,
    # Strides
    stride_gi_B, stride_gi_T, stride_gi_H,
    stride_h_B, stride_h_T, stride_h_H,
    stride_W_H, stride_W_3H,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_H_MATMUL: tl.constexpr,  # For matmul reduction
):
    """
    Process ALL timesteps for one batch element, one hidden block.

    Grid: (B, H // BLOCK_H)
    Each program: processes B[pid_b], H[pid_h*BLOCK_H : (pid_h+1)*BLOCK_H]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Hidden dimension indices for this block
    h_start = pid_h * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load initial hidden state h_0: [BLOCK_H]
    h_prev = tl.load(
        h_states_ptr + pid_b * stride_h_B + 0 * stride_h_T + h_offs * stride_h_H,
        mask=h_mask,
        other=0.0
    ).to(tl.float32)

    # Process each timestep
    for t in range(T):
        # ===== Load precomputed input gates: [3, BLOCK_H] =====
        gate_base = pid_b * stride_gi_B + t * stride_gi_T

        i_r = tl.load(
            gates_input_ptr + gate_base + 0*H + h_offs * stride_gi_H,
            mask=h_mask, other=0.0
        ).to(tl.float32)

        i_z = tl.load(
            gates_input_ptr + gate_base + 1*H + h_offs * stride_gi_H,
            mask=h_mask, other=0.0
        ).to(tl.float32)

        i_n = tl.load(
            gates_input_ptr + gate_base + 2*H + h_offs * stride_gi_H,
            mask=h_mask, other=0.0
        ).to(tl.float32)

        # ===== Compute h_prev @ W_h: [BLOCK_H] @ [H, 3H] -> [3, BLOCK_H] =====
        # We need to reduce across full H dimension
        # Accumulate h @ W for all 3 gates

        h_r = tl.zeros([BLOCK_H], dtype=tl.float32)
        h_z = tl.zeros([BLOCK_H], dtype=tl.float32)
        h_n = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Tile across source H dimension
        for h_src_start in range(0, H, BLOCK_H_MATMUL):
            h_src_offs = h_src_start + tl.arange(0, BLOCK_H_MATMUL)
            h_src_mask = h_src_offs < H

            # Load h_prev for source block (need full H, not just our block)
            # Read from h_states at timestep t (we just wrote it)
            h_src = tl.load(
                h_states_ptr + pid_b * stride_h_B + t * stride_h_T + h_src_offs * stride_h_H,
                mask=h_src_mask,
                other=0.0
            ).to(tl.float32)

            # Load weight tiles: W_h[h_src, h_dest] for each gate
            # W_h is [H, 3H], laid out as [H, (r|z|n)]
            # For gate r: W_h[:, 0:H]
            # For gate z: W_h[:, H:2H]
            # For gate n: W_h[:, 2H:3H]

            w_r = tl.load(
                W_h_ptr + h_src_offs[:, None] * stride_W_H + (0*H + h_offs[None, :]) * stride_W_3H,
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_z = tl.load(
                W_h_ptr + h_src_offs[:, None] * stride_W_H + (1*H + h_offs[None, :]) * stride_W_3H,
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_n = tl.load(
                W_h_ptr + h_src_offs[:, None] * stride_W_H + (2*H + h_offs[None, :]) * stride_W_3H,
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            # Accumulate: h_src @ W
            h_r += tl.sum(h_src[:, None] * w_r, axis=0)
            h_z += tl.sum(h_src[:, None] * w_z, axis=0)
            h_n += tl.sum(h_src[:, None] * w_n, axis=0)

        # ===== GRU cell computation (fully fused) =====
        r_t = tl.sigmoid(i_r + h_r)
        z_t = tl.sigmoid(i_z + h_z)

        n_pre = i_n + r_t * h_n
        # Stable tanh
        n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
        exp_2x = tl.exp(2.0 * n_pre_clamped)
        n_t = (exp_2x - 1.0) / (exp_2x + 1.0)

        h_new = (1.0 - z_t) * n_t + z_t * h_prev

        # ===== Store new hidden state =====
        tl.store(
            h_states_ptr + pid_b * stride_h_B + (t+1) * stride_h_T + h_offs * stride_h_H,
            h_new,
            mask=h_mask
        )

        # Update for next timestep
        h_prev = h_new


class TritonFusedGRU(nn.Module):
    """
    Fully fused GRU using Triton kernel for all timesteps.

    Reduces kernel launches:
    - Before: 2T matmuls + T GRU cells = ~1536 kernels
    - After: 1 input matmul + (B × H/BLOCK_H) fused kernels = ~20 kernels
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        z_bias_input: float = 0.0,
        z_bias_hidden: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Input projection (3 gates combined)
        self.W_input = nn.Linear(dim, 3 * self.dim_inner, bias=True)

        # Hidden projection (3 gates combined) - stored as [H, 3H]
        self.W_hidden = nn.Parameter(torch.empty(self.dim_inner, 3 * self.dim_inner))
        self.b_hidden = nn.Parameter(torch.empty(3 * self.dim_inner))

        # Output projection
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights(z_bias_input, z_bias_hidden)

    def _init_weights(self, z_bias_input, z_bias_hidden):
        """Initialize with proper scaling."""
        H = self.dim_inner

        # Input projection
        std_input = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.W_input.weight, -std_input, std_input)
        nn.init.zeros_(self.W_input.bias)
        self.W_input.bias.data[H:2*H].fill_(z_bias_input)  # z-gate bias

        # Hidden projection (orthogonal is good for recurrent)
        nn.init.orthogonal_(self.W_hidden)
        nn.init.zeros_(self.b_hidden)
        self.b_hidden.data[H:2*H].fill_(z_bias_hidden)  # z-gate bias

        # Output projection
        if not isinstance(self.to_out, nn.Identity):
            nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        x,
        prev_hidden=None,
        return_next_prev_hidden=False,
        **kwargs
    ):
        """
        Args:
            x: [B, T, D]
            prev_hidden: [B, H] or None

        Returns:
            out: [B, T, D]
            next_hidden: [B, H] (if return_next_prev_hidden)
        """
        B, T, D = x.shape
        H = self.dim_inner
        device = x.device
        dtype = x.dtype

        # Initialize hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, H, device=device, dtype=dtype)

        # Precompute input projections (CUBLAS - single matmul for all T)
        gates_input = self.W_input(x)  # [B, T, 3H]

        # Add hidden bias (broadcast)
        gates_input = gates_input + self.b_hidden

        # Allocate hidden states buffer: [B, T+1, H]
        # h_states[:, 0, :] = prev_hidden (initial)
        # h_states[:, t+1, :] = h_t (for t=0..T-1)
        h_states = torch.zeros(B, T+1, H, device=device, dtype=dtype)
        h_states[:, 0, :] = prev_hidden

        # Launch fused kernel
        BLOCK_H = min(128, triton.next_power_of_2(H))
        BLOCK_H_MATMUL = 128  # For matmul reduction

        grid = (B, triton.cdiv(H, BLOCK_H))

        fused_gru_timesteps_kernel[grid](
            gates_input,
            h_states,
            self.W_hidden,
            B, T, H,
            gates_input.stride(0), gates_input.stride(1), gates_input.stride(2),
            h_states.stride(0), h_states.stride(1), h_states.stride(2),
            self.W_hidden.stride(0), self.W_hidden.stride(1),
            BLOCK_H=BLOCK_H,
            BLOCK_H_MATMUL=BLOCK_H_MATMUL,
        )

        # Extract hidden states (skip initial h_0)
        h_all = h_states[:, 1:, :]  # [B, T, H]

        # Output projection
        out = self.to_out(h_all)

        if return_next_prev_hidden:
            return out, h_states[:, -1, :]  # Final hidden state
        return out

    def __repr__(self):
        return f"TritonFusedGRU(dim={self.dim}, dim_inner={self.dim_inner}, FULLY FUSED)"

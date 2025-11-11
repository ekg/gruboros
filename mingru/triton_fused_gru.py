"""
Fully Fused Triton GRU - Single Kernel Per Layer

Fuses ALL operations for standard GRU into one kernel:
- Input matmuls (3 gates)
- Hidden matmuls (3 gates)
- All activations (sigmoid, tanh)
- Hidden state updates
- Sequential timestep loop (done in kernel, not Python)

Reduces from 1,536 kernel launches (3 per timestep) to 1 per layer.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gru_forward_kernel(
    # Input: [B, T, D]
    x_ptr,
    # Weights: [D, H] for each gate
    W_ir_ptr, W_iz_ptr, W_in_ptr,  # Input weights (reset, update, new)
    W_hr_ptr, W_hz_ptr, W_hn_ptr,  # Hidden weights
    # Biases: [H] for each gate (optional, can be None)
    b_ir_ptr, b_iz_ptr, b_in_ptr,
    b_hr_ptr, b_hz_ptr, b_hn_ptr,
    # Hidden states: [B, H] (input and output)
    h_prev_ptr, h_out_ptr,
    # All hidden states for all timesteps: [B, T, H]
    h_all_ptr,
    # Dimensions
    B, T, D, H,
    # Strides
    stride_xB, stride_xT, stride_xD,
    stride_hB, stride_hH,
    stride_hallB, stride_hallT, stride_hallH,
    # Block sizes
    BLOCK_H: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """
    Single kernel that processes entire sequence for one batch element.

    Each program handles:
    - 1 batch element
    - 1 block of hidden dims (BLOCK_H)
    - ALL timesteps (sequential in kernel)

    Grid: (B, cdiv(H, BLOCK_H))
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_h = tl.program_id(1)  # Hidden dim block

    # Hidden dimension indices for this block
    h_start = pid_h * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load initial hidden state: [BLOCK_H]
    h_prev_offset = pid_b * stride_hB + h_offs * stride_hH
    h_t = tl.load(h_prev_ptr + h_prev_offset, mask=h_mask, other=0.0).to(tl.float32)

    # Process each timestep sequentially
    for t in range(T):
        # ===== Load input for timestep t: [D] =====
        # We need to do matmul: x[t] @ W, so we need full D dimension
        # Load x[t] for all D dims
        x_offset = pid_b * stride_xB + t * stride_xT

        # Compute input contributions for all 3 gates
        # i_r = x[t] @ W_ir, i_z = x[t] @ W_iz, i_n = x[t] @ W_in
        # We need to accumulate across D dimension

        i_r = tl.zeros([BLOCK_H], dtype=tl.float32)
        i_z = tl.zeros([BLOCK_H], dtype=tl.float32)
        i_n = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Tile across D dimension for matmul
        D_BLOCK = 128  # Process D in blocks
        for d_start in range(0, D, D_BLOCK):
            d_offs = d_start + tl.arange(0, D_BLOCK)
            d_mask = d_offs < D

            # Load x[t, d_offs]: [D_BLOCK]
            x_tile = tl.load(
                x_ptr + x_offset + d_offs * stride_xD,
                mask=d_mask,
                other=0.0
            ).to(tl.float32)

            # Load weight columns: [D_BLOCK, BLOCK_H]
            # W is [D, H], we want W[d_offs, h_offs]
            w_r_tile = tl.load(
                W_ir_ptr + d_offs[:, None] * H + h_offs[None, :],
                mask=d_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_z_tile = tl.load(
                W_iz_ptr + d_offs[:, None] * H + h_offs[None, :],
                mask=d_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_n_tile = tl.load(
                W_in_ptr + d_offs[:, None] * H + h_offs[None, :],
                mask=d_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            # Accumulate matmul: x @ W (dot product)
            i_r += tl.sum(x_tile[:, None] * w_r_tile, axis=0)
            i_z += tl.sum(x_tile[:, None] * w_z_tile, axis=0)
            i_n += tl.sum(x_tile[:, None] * w_n_tile, axis=0)

        # ===== Compute hidden contributions: h_{t-1} @ W_h =====
        h_r = tl.zeros([BLOCK_H], dtype=tl.float32)
        h_z = tl.zeros([BLOCK_H], dtype=tl.float32)
        h_n = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Tile across H dimension for matmul (h_t is [H])
        H_BLOCK = 128
        for h_src_start in range(0, H, H_BLOCK):
            h_src_offs = h_src_start + tl.arange(0, H_BLOCK)
            h_src_mask = h_src_offs < H

            # Load h_{t-1}[h_src_offs] - need to reload from global memory
            # Actually, we have h_t in register for current block, but need full H
            # This is a problem - we need ALL of h_{t-1} but only have BLOCK_H
            # Solution: Use shared memory or make H small enough
            # For now: reload from global memory (prev iteration wrote it)
            if t == 0:
                h_src_tile = tl.load(
                    h_prev_ptr + pid_b * stride_hB + h_src_offs * stride_hH,
                    mask=h_src_mask,
                    other=0.0
                ).to(tl.float32)
            else:
                # Load from h_all_ptr at t-1
                h_src_tile = tl.load(
                    h_all_ptr + pid_b * stride_hallB + (t-1) * stride_hallT + h_src_offs * stride_hallH,
                    mask=h_src_mask,
                    other=0.0
                ).to(tl.float32)

            # Load weight columns for hidden matmul: [H_src, BLOCK_H]
            w_hr_tile = tl.load(
                W_hr_ptr + h_src_offs[:, None] * H + h_offs[None, :],
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_hz_tile = tl.load(
                W_hz_ptr + h_src_offs[:, None] * H + h_offs[None, :],
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            w_hn_tile = tl.load(
                W_hn_ptr + h_src_offs[:, None] * H + h_offs[None, :],
                mask=h_src_mask[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float32)

            # Accumulate
            h_r += tl.sum(h_src_tile[:, None] * w_hr_tile, axis=0)
            h_z += tl.sum(h_src_tile[:, None] * w_hz_tile, axis=0)
            h_n += tl.sum(h_src_tile[:, None] * w_hn_tile, axis=0)

        # ===== Add biases (if enabled) =====
        if USE_BIAS:
            b_ir_tile = tl.load(b_ir_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            b_iz_tile = tl.load(b_iz_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            b_in_tile = tl.load(b_in_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            b_hr_tile = tl.load(b_hr_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            b_hz_tile = tl.load(b_hz_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            b_hn_tile = tl.load(b_hn_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)

            i_r += b_ir_tile
            i_z += b_iz_tile
            i_n += b_in_tile
            h_r += b_hr_tile
            h_z += b_hz_tile
            h_n += b_hn_tile

        # ===== GRU activations =====
        # r_t = sigmoid(i_r + h_r)
        r_t = tl.sigmoid(i_r + h_r)

        # z_t = sigmoid(i_z + h_z)
        z_t = tl.sigmoid(i_z + h_z)

        # n_t = tanh(i_n + r_t * h_n)
        n_pre = i_n + r_t * h_n
        # Numerically stable tanh
        n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
        exp_2x = tl.exp(2.0 * n_pre_clamped)
        n_t = (exp_2x - 1.0) / (exp_2x + 1.0)

        # ===== Update hidden state =====
        # h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        h_t = (1.0 - z_t) * n_t + z_t * h_t

        # ===== Store hidden state for this timestep =====
        h_out_offset = pid_b * stride_hallB + t * stride_hallT + h_offs * stride_hallH
        tl.store(h_all_ptr + h_out_offset, h_t, mask=h_mask)

    # Store final hidden state
    tl.store(h_out_ptr + h_prev_offset, h_t, mask=h_mask)


class TritonFusedGRU(nn.Module):
    """
    Fully fused Triton GRU - processes entire layer in ONE kernel launch.

    Comparison:
    - HybridFusedGRU: 1,536 kernel launches (3 per timestep × 512 timesteps)
    - TritonFusedGRU: 1 kernel launch

    Trade-off: More complex kernel, but eliminates kernel launch overhead.
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
        self.z_bias_input = z_bias_input
        self.z_bias_hidden = z_bias_hidden

        # Input projection weights: [dim, dim_inner] for each gate
        self.W_ir = nn.Parameter(torch.empty(dim, self.dim_inner))
        self.W_iz = nn.Parameter(torch.empty(dim, self.dim_inner))
        self.W_in = nn.Parameter(torch.empty(dim, self.dim_inner))

        # Hidden projection weights: [dim_inner, dim_inner] for each gate
        self.W_hr = nn.Parameter(torch.empty(self.dim_inner, self.dim_inner))
        self.W_hz = nn.Parameter(torch.empty(self.dim_inner, self.dim_inner))
        self.W_hn = nn.Parameter(torch.empty(self.dim_inner, self.dim_inner))

        # Biases: [dim_inner] for each gate
        self.b_ir = nn.Parameter(torch.empty(self.dim_inner))
        self.b_iz = nn.Parameter(torch.empty(self.dim_inner))
        self.b_in = nn.Parameter(torch.empty(self.dim_inner))
        self.b_hr = nn.Parameter(torch.empty(self.dim_inner))
        self.b_hz = nn.Parameter(torch.empty(self.dim_inner))
        self.b_hn = nn.Parameter(torch.empty(self.dim_inner))

        # Output projection (if expanding)
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        std_input = 1.0 / math.sqrt(self.dim)
        std_hidden = 1.0 / math.sqrt(self.dim_inner)

        # Input weights
        for w in [self.W_ir, self.W_iz, self.W_in]:
            nn.init.uniform_(w, -std_input, std_input)

        # Hidden weights (use orthogonal for recurrent)
        for w in [self.W_hr, self.W_hz, self.W_hn]:
            nn.init.orthogonal_(w)

        # Biases
        nn.init.zeros_(self.b_ir)
        nn.init.constant_(self.b_iz, self.z_bias_input)  # Update gate bias
        nn.init.zeros_(self.b_in)
        nn.init.zeros_(self.b_hr)
        nn.init.constant_(self.b_hz, self.z_bias_hidden)  # Update gate bias
        nn.init.zeros_(self.b_hn)

        # Output projection
        if not isinstance(self.to_out, nn.Identity):
            nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        x,
        prev_hidden=None,
        return_next_prev_hidden=False,
        **kwargs  # Ignore extra args for compatibility
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

        # Output buffers
        h_all = torch.empty(B, T, H, device=device, dtype=dtype)
        h_final = torch.empty(B, H, device=device, dtype=dtype)

        # Grid: (batch, hidden_blocks)
        BLOCK_H = min(128, triton.next_power_of_2(H))
        grid = (B, triton.cdiv(H, BLOCK_H))

        # Launch fused kernel
        fused_gru_forward_kernel[grid](
            x,
            self.W_ir, self.W_iz, self.W_in,
            self.W_hr, self.W_hz, self.W_hn,
            self.b_ir, self.b_iz, self.b_in,
            self.b_hr, self.b_hz, self.b_hn,
            prev_hidden, h_final, h_all,
            B, T, D, H,
            x.stride(0), x.stride(1), x.stride(2),
            prev_hidden.stride(0), prev_hidden.stride(1),
            h_all.stride(0), h_all.stride(1), h_all.stride(2),
            BLOCK_H=BLOCK_H,
            USE_BIAS=True,
        )

        # Output projection
        out = self.to_out(h_all)

        if return_next_prev_hidden:
            return out, h_final
        return out

    def __repr__(self):
        return f"TritonFusedGRU(dim={self.dim}, dim_inner={self.dim_inner}, SINGLE FUSED KERNEL)"

"""
Persistent-over-time GRU using Triton.

Key ideas:
- Precompute X @ W_* via cuBLAS (fast batched GEMM)
- Triton kernel: one launch per (batch_tile, hidden_tile)
- Loop over T inside kernel, keeping h in registers
- Use H_curr scratch buffer for cross-tile communication
- Tile h @ U using tl.dot with K-loop accumulation

Based on guidance for tiling h @ U properly with persistent-T pattern.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gru_persistent_forward_kernel(
    # Precomputed input matmuls [T, B, H]
    XW_z_ptr, XW_r_ptr, XW_n_ptr,
    # Packed recurrent weights [3, H, H] (z=0, r=1, n=2)
    U_ptr,
    # Biases [H]
    b_z_ptr, b_r_ptr, b_n_ptr,
    # Initial hidden [B, H]
    H0_ptr,
    # Scratch: current full hidden [B, H] for cross-tile reading
    H_curr_ptr,
    # Scratch: current r gate [B, H] for n computation
    R_curr_ptr,
    # Output [T, B, H]
    H_out_ptr,
    # Optional document boundaries [T, B] (0/1)
    mask_reset_ptr,
    # Dimensions
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    # Strides for XW [T, B, H]
    stride_xw_t: tl.constexpr,
    stride_xw_b: tl.constexpr,
    stride_xw_h: tl.constexpr,
    # Strides for U [3, H, H] (gate, k, n)
    stride_u_g: tl.constexpr,
    stride_u_k: tl.constexpr,
    stride_u_n: tl.constexpr,
    # Strides for H0 [B, H]
    stride_h0_b: tl.constexpr,
    stride_h0_h: tl.constexpr,
    # Strides for H_out [T, B, H]
    stride_out_t: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_h: tl.constexpr,
    # Config
    have_mask: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Persistent GRU kernel - one program per (batch, hidden_tile) across all T.

    Grid: (B, ceil_div(H, BLOCK_H))
    """
    # Program IDs
    b = tl.program_id(0)
    h_block = tl.program_id(1)

    # Hidden N-tile offsets
    h_n_offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_n_mask = h_n_offs < H

    # Load initial hidden state for this tile
    h = tl.load(H0_ptr + b * stride_h0_b + h_n_offs * stride_h0_h,
                mask=h_n_mask, other=0.0).to(tl.float32)

    # Load biases for this tile
    bz = tl.load(b_z_ptr + h_n_offs, mask=h_n_mask, other=0.0).to(tl.float32)
    br = tl.load(b_r_ptr + h_n_offs, mask=h_n_mask, other=0.0).to(tl.float32)
    bn = tl.load(b_n_ptr + h_n_offs, mask=h_n_mask, other=0.0).to(tl.float32)

    # Write initial h to H_curr so other tiles can read it
    tl.store(H_curr_ptr + b * H + h_n_offs, h, mask=h_n_mask)

    # Main persistent time loop
    for t in range(T):
        # === Document boundary reset ===
        if have_mask:
            reset_flag = tl.load(mask_reset_ptr + t * B + b).to(tl.float32)
            h = h * (1.0 - reset_flag)  # Zero out if reset_flag=1

        # === Load input contributions (XW) ===
        base_tb = t * stride_xw_t + b * stride_xw_b
        xwz = tl.load(XW_z_ptr + base_tb + h_n_offs * stride_xw_h,
                      mask=h_n_mask, other=0.0).to(tl.float32) + bz
        xwr = tl.load(XW_r_ptr + base_tb + h_n_offs * stride_xw_h,
                      mask=h_n_mask, other=0.0).to(tl.float32) + br
        xwn = tl.load(XW_n_ptr + base_tb + h_n_offs * stride_xw_h,
                      mask=h_n_mask, other=0.0).to(tl.float32) + bn

        # === Compute z and r gates: h @ U (tiled over K) ===
        acc_z = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_r = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k0 in range(0, H, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < H

            # Load K-tile of current hidden from H_curr [BLOCK_K]
            h_k = tl.load(H_curr_ptr + b * H + k_offs,
                         mask=k_mask, other=0.0).to(tl.float32)

            # Load U tiles for z and r gates [BLOCK_K, BLOCK_H]
            Uz = tl.load(U_ptr + 0 * stride_u_g +
                        k_offs[:, None] * stride_u_k +
                        h_n_offs[None, :] * stride_u_n,
                        mask=k_mask[:, None] & h_n_mask[None, :],
                        other=0.0).to(tl.float32)

            Ur = tl.load(U_ptr + 1 * stride_u_g +
                        k_offs[:, None] * stride_u_k +
                        h_n_offs[None, :] * stride_u_n,
                        mask=k_mask[:, None] & h_n_mask[None, :],
                        other=0.0).to(tl.float32)

            # GEMV: manually accumulate h_k @ U
            # For each output element j, compute sum_i h_k[i] * U[i, j]
            acc_z += tl.sum(h_k[:, None] * Uz, axis=0)  # [BLOCK_K, BLOCK_H] -> [BLOCK_H]
            acc_r += tl.sum(h_k[:, None] * Ur, axis=0)

        # Compute z and r
        z = tl.sigmoid(xwz + acc_z)
        r = tl.sigmoid(xwr + acc_r)

        # Write r to R_curr for use in n computation
        tl.store(R_curr_ptr + b * H + h_n_offs, r, mask=h_n_mask)

        # === Compute n gate: (r * h) @ U_n (tiled over K) ===
        acc_n = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k0 in range(0, H, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < H

            # Load K-tiles of r and h
            r_k = tl.load(R_curr_ptr + b * H + k_offs,
                         mask=k_mask, other=0.0).to(tl.float32)
            h_k = tl.load(H_curr_ptr + b * H + k_offs,
                         mask=k_mask, other=0.0).to(tl.float32)

            # Compute r * h element-wise
            rh_k = r_k * h_k

            # Load U_n tile [BLOCK_K, BLOCK_H]
            Un = tl.load(U_ptr + 2 * stride_u_g +
                        k_offs[:, None] * stride_u_k +
                        h_n_offs[None, :] * stride_u_n,
                        mask=k_mask[:, None] & h_n_mask[None, :],
                        other=0.0).to(tl.float32)

            # GEMV: manually accumulate (r*h) @ U_n
            acc_n += tl.sum(rh_k[:, None] * Un, axis=0)  # [BLOCK_K, BLOCK_H] -> [BLOCK_H]

        # Compute n with tanh
        n_pre = xwn + acc_n
        # Manual tanh
        n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
        exp_2x = tl.exp(2.0 * n_pre_clamped)
        n = (exp_2x - 1.0) / (exp_2x + 1.0)

        # === GRU update ===
        h = (1.0 - z) * n + z * h

        # Write output for this timestep
        tl.store(H_out_ptr + t * stride_out_t + b * stride_out_b + h_n_offs * stride_out_h,
                h, mask=h_n_mask)

        # Update H_curr for next timestep
        tl.store(H_curr_ptr + b * H + h_n_offs, h, mask=h_n_mask)


class PersistentGRU(nn.Module):
    """
    GRU with persistent-over-time Triton kernel.

    Target: T=512, B=90, H=2048, dtype=bfloat16
    """

    def __init__(self, dim, expansion_factor=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Input projection: dim -> 3*dim_inner (for z, r, n gates)
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=False)

        # Recurrent projection: dim_inner -> 3*dim_inner (for z, r, n gates)
        # Pack as [3, H, H] for efficient access
        self.U_z = nn.Linear(self.dim_inner, self.dim_inner, bias=False)
        self.U_r = nn.Linear(self.dim_inner, self.dim_inner, bias=False)
        self.U_n = nn.Linear(self.dim_inner, self.dim_inner, bias=False)

        # Biases
        self.bias_z = nn.Parameter(torch.zeros(self.dim_inner))
        self.bias_r = nn.Parameter(torch.zeros(self.dim_inner))
        self.bias_n = nn.Parameter(torch.zeros(self.dim_inner))

        # Output projection
        self.output_projection = nn.Linear(self.dim_inner, dim, bias=False)

    def _pack_U(self):
        """Pack U weights as [3, H, H] for kernel."""
        U_z = self.U_z.weight.t().contiguous()  # [H, H]
        U_r = self.U_r.weight.t().contiguous()
        U_n = self.U_n.weight.t().contiguous()
        return torch.stack([U_z, U_r, U_n], dim=0)  # [3, H, H]

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

        # Step 1: Precompute input gates via cuBLAS [B, T, 3*H]
        input_gates = self.input_projection(x)  # [B, T, 3*H]

        # Reshape to [T, B, H] for each gate
        input_gates = input_gates.transpose(0, 1)  # [T, B, 3*H]
        XW_z = input_gates[:, :, :H].contiguous()
        XW_r = input_gates[:, :, H:2*H].contiguous()
        XW_n = input_gates[:, :, 2*H:].contiguous()

        # Step 2: Pack U weights
        U_packed = self._pack_U()  # [3, H, H]

        # Step 3: Prepare initial hidden
        if prev_hiddens is None:
            h0 = torch.zeros(B, H, device=device, dtype=torch.float32)
        else:
            h0 = prev_hiddens.to(torch.float32)

        # Step 4: Allocate scratch and output
        H_curr = torch.empty(B, H, device=device, dtype=torch.float32)
        R_curr = torch.empty(B, H, device=device, dtype=torch.float32)
        H_out = torch.empty(T, B, H, device=device, dtype=torch.float32)

        # Step 5: Prepare document boundaries
        if doc_boundaries is not None:
            # Transpose to [T, B] and convert to float
            mask_reset = doc_boundaries.t().to(torch.float32).contiguous()
        else:
            mask_reset = None

        # Step 6: Launch persistent kernel
        BLOCK_H = 128
        BLOCK_K = 64
        grid = (B, triton.cdiv(H, BLOCK_H))

        gru_persistent_forward_kernel[grid](
            XW_z, XW_r, XW_n,
            U_packed,
            self.bias_z, self.bias_r, self.bias_n,
            h0,
            H_curr,
            R_curr,
            H_out,
            mask_reset,
            T, B, H,
            # XW strides [T, B, H]
            XW_z.stride(0), XW_z.stride(1), XW_z.stride(2),
            # U strides [3, H, H]
            U_packed.stride(0), U_packed.stride(1), U_packed.stride(2),
            # h0 strides [B, H]
            h0.stride(0), h0.stride(1),
            # H_out strides [T, B, H]
            H_out.stride(0), H_out.stride(1), H_out.stride(2),
            have_mask=(mask_reset is not None),
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        # Step 7: Project output
        H_out_transposed = H_out.transpose(0, 1)  # [B, T, H]
        output = self.output_projection(H_out_transposed.to(dtype))  # [B, T, D]

        # Final hidden is last timestep
        h_final = H_out[-1].to(dtype)  # [B, H]

        if return_next_prev_hidden:
            return output, h_final
        else:
            return output

    def __repr__(self):
        return f"PersistentGRU(dim={self.dim}, dim_inner={self.dim_inner}, persistent-T Triton)"

"""
Low-Rank GRU: Factor recurrent weights to reduce FLOPs.

Architecture:
    Factor U3[H, 3H] → A[H, R] · B[R, 3H]

    Forward pass:
        gates = h @ U3                    # Original: (B×H)·(H×3H) = 3B·H² FLOPs
        gates = (h @ A) @ B               # Low-rank: B·H·R + B·R·3H = 4B·H·R FLOPs

Key optimization:
- No change to input/output dimensions (unlike ProjectedGRU)
- Only change: recurrent matmul split into two smaller matmuls
- FLOPs reduction: 3B·H² → 4B·H·R
  - R=256: 6× less (for H=2048)
  - R=384: 4× less
  - R=512: 3× less

Trade-off: Lower R = faster but may lose capacity
"""

import torch
import torch.nn as nn
from mingru.hybrid_fused_gru import gru_cell_fused  # Import Triton fused cell
import triton
import triton.language as tl
import math


class LowRankGRU(nn.Module):
    """
    GRU with low-rank factorized recurrent weights.

    Factorizes U3[H, 3H] → A[H, R] · B[R, 3H]
    Two matmuls per timestep instead of one large matmul.

    Args:
        dim: Model dimension (H)
        rank: Factorization rank (R < H)
        expansion_factor: Expansion for GRU inner dim (usually 1.0)
    """

    def __init__(
        self,
        dim: int,
        rank: int = 384,
        expansion_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.rank = rank

        if self.rank > self.dim_inner:
            raise ValueError(f"rank ({self.rank}) must be <= dim_inner ({self.dim_inner})")

        # Input projection: x → 3H (same as baseline)
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=True)

        # Factorized recurrent weights: U3[H, 3H] → A[H, R] · B[R, 3H]
        self.A = nn.Linear(self.dim_inner, self.rank, bias=False)  # h @ A: (B×H)·(H×R)
        self.B = nn.Linear(self.rank, 3 * self.dim_inner, bias=False)  # temp @ B: (B×R)·(R×3H)

        # Output projection (identity for now, matching HybridFusedGRU)
        self.to_out = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: torch.Tensor = None,
        return_next_prev_hidden: bool = False,
        doc_boundaries: torch.Tensor = None,
    ):
        """
        Args:
            x: [B, T, D]
            prev_hidden: [B, H]
            return_next_prev_hidden: bool
            doc_boundaries: [B, T] bool

        Returns:
            output: [B, T, D]
            h_final: [B, H] (if return_next_prev_hidden)
        """
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden state
        if prev_hidden is None:
            h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            h = prev_hidden

        # Precompute input gates: [B, T, 3H]
        input_gates = self.input_projection(x)

        # Document boundary mask
        if doc_boundaries is not None:
            doc_mask = doc_boundaries  # [B, T]
        else:
            doc_mask = None

        # Recurrent loop over timesteps
        outputs = []
        BLOCK_SIZE = 2 ** math.ceil(math.log2(self.dim_inner))  # Next power of 2

        for t in range(T):
            # Reset hidden state at document boundaries
            if doc_mask is not None:
                reset = doc_mask[:, t]  # [B]
                if reset.any():
                    h = h.masked_fill(reset.unsqueeze(-1), 0.0)

            # Input gates for this timestep: [B, 3H]
            i_gates = input_gates[:, t, :].contiguous()

            # Low-rank recurrent matmul: (h @ A) @ B
            # Step 1: h @ A → [B, R]
            temp = self.A(h)  # (B×H) @ (H×R) → (B×R)

            # Step 2: temp @ B → [B, 3H]
            h_gates = self.B(temp).contiguous()  # (B×R) @ (R×3H) → (B×3H)

            # Use Triton fused cell for GRU update (CUDA only)
            if device.type == 'cuda':
                # Allocate fp32 buffer for Triton kernel output
                h_new_fp32 = torch.empty(B, self.dim_inner, device=device, dtype=torch.float32)

                # Launch Triton kernel for fused cell computation
                grid = (B,)
                gru_cell_fused[grid](
                    i_gates, h_gates,
                    h, h_new_fp32,
                    B, self.dim_inner,
                    BLOCK_SIZE
                )
                h = h_new_fp32.to(dtype)
            else:
                # CPU fallback (pure PyTorch)
                H = self.dim_inner
                i_r, i_z, i_n = i_gates.split(H, dim=1)
                gh_r, gh_z, gh_n = h_gates.split(H, dim=1)
                r = torch.sigmoid(i_r + gh_r)
                z = torch.sigmoid(i_z + gh_z)
                n = torch.tanh(i_n + r * gh_n)
                h = (1.0 - z) * h + z * n

            outputs.append(h)

        # Stack outputs: [B, T, H]
        output = torch.stack(outputs, dim=1)

        # Output projection
        output = self.to_out(output)

        if return_next_prev_hidden:
            return output, h
        else:
            return output

    def __repr__(self):
        flops_original = 3 * self.dim_inner ** 2
        flops_lowrank = 4 * self.dim_inner * self.rank
        reduction = flops_original / flops_lowrank
        return (
            f"LowRankGRU(dim={self.dim}, rank={self.rank}, "
            f"reduction={reduction:.1f}× FLOPs)"
        )


class LowRankGRUFused(nn.Module):
    """
    LowRankGRU with Triton-fused cell updates.

    Same low-rank factorization, but uses the fused cell kernel from HybridFusedGRU
    for the pointwise ops (sigmoid, tanh, GRU update).

    This should be faster than the PyTorch loop version above.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 384,
        expansion_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.rank = rank

        if self.rank > self.dim_inner:
            raise ValueError(f"rank ({self.rank}) must be <= dim_inner ({self.dim_inner})")

        # Input projection
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=True)

        # Factorized recurrent weights
        self.A = nn.Linear(self.dim_inner, self.rank, bias=False)
        self.B = nn.Linear(self.rank, 3 * self.dim_inner, bias=False)

        # Output projection
        self.to_out = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: torch.Tensor = None,
        return_next_prev_hidden: bool = False,
        doc_boundaries: torch.Tensor = None,
    ):
        """Forward with fused cell kernel (TODO: port from HybridFusedGRU)."""
        # For now, fall back to unfused version
        # In production, would port the Triton fused_gru_cell_kernel here
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        if prev_hidden is None:
            h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            h = prev_hidden

        input_gates = self.input_projection(x)

        if doc_boundaries is not None:
            doc_mask = doc_boundaries
        else:
            doc_mask = None

        outputs = []
        for t in range(T):
            if doc_mask is not None:
                reset = doc_mask[:, t:t+1]
                h = h * (1.0 - reset.to(dtype))

            i_gates = input_gates[:, t, :]

            # Low-rank factorization
            temp = self.A(h)
            h_gates = self.B(temp)

            # Fused cell update (TODO: use Triton kernel)
            H = self.dim_inner
            i_r, i_z, i_n = i_gates.split(H, dim=1)
            gh_r, gh_z, gh_n = h_gates.split(H, dim=1)

            r = torch.sigmoid(i_r + gh_r)
            z = torch.sigmoid(i_z + gh_z)
            n = torch.tanh(i_n + r * gh_n)
            h = (1.0 - z) * h + z * n

            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        output = self.to_out(output)

        if return_next_prev_hidden:
            return output, h
        else:
            return output

    def __repr__(self):
        flops_original = 3 * self.dim_inner ** 2
        flops_lowrank = 4 * self.dim_inner * self.rank
        reduction = flops_original / flops_lowrank
        return (
            f"LowRankGRUFused(dim={self.dim}, rank={self.rank}, "
            f"reduction={reduction:.1f}× FLOPs)"
        )

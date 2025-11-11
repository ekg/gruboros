"""
Projected GRU: Reduce recurrent dimension for faster updates.

Architecture:
    Input [B, T, D]
    → project_down: Linear(D, H_rec)
    → HybridFusedGRU(H_rec)  # Smaller recurrent matmul!
    → project_up: Linear(H_rec, D)
    → Output [B, T, D]

Key optimization:
- Recurrent matmul is (B×H_rec) @ (H_rec×3H_rec) instead of (B×D) @ (D×3D)
- FLOPs: 3B·H_rec² vs 3B·D² (4× less if H_rec = D/2)
- Similar to projected LSTM, no accuracy loss if H_rec chosen well

Recommended: H_rec = D/2 (e.g., 1024 for D=2048)
"""

import torch
import torch.nn as nn
from mingru.hybrid_fused_gru import HybridFusedGRU


class ProjectedGRU(nn.Module):
    """
    GRU with projection to reduce recurrent dimension.

    Wraps HybridFusedGRU with down/up projection layers.
    No new kernels needed - just smaller matmuls in the recurrent step.

    Args:
        dim: Model dimension (D)
        h_recurrent: Recurrent dimension (H_rec < D)
        expansion_factor: Expansion for GRU inner dim (usually 1.0)
    """

    def __init__(
        self,
        dim: int,
        h_recurrent: int = None,
        expansion_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.h_recurrent = h_recurrent or (dim // 2)  # Default: half

        if self.h_recurrent > dim:
            raise ValueError(f"h_recurrent ({self.h_recurrent}) must be <= dim ({dim})")

        # Project down: [B, T, D] → [B, T, H_rec]
        self.project_down = nn.Linear(dim, self.h_recurrent, bias=False)

        # Core recurrent GRU (operates on smaller H_rec space)
        self.gru = HybridFusedGRU(
            dim=self.h_recurrent,
            expansion_factor=expansion_factor,
            **kwargs
        )

        # Project up: [B, T, H_rec] → [B, T, D]
        self.project_up = nn.Linear(self.h_recurrent, dim, bias=False)

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
            prev_hidden: [B, H_rec] (note: smaller than D!)
            return_next_prev_hidden: bool
            doc_boundaries: [B, T] bool

        Returns:
            output: [B, T, D]
            h_final: [B, H_rec] (if return_next_prev_hidden)
        """
        # Project down to recurrent space
        x_rec = self.project_down(x)  # [B, T, D] → [B, T, H_rec]

        # Run GRU in smaller space (this is where speedup happens!)
        if return_next_prev_hidden:
            h_rec, h_final_rec = self.gru(
                x_rec,
                prev_hidden=prev_hidden,
                return_next_prev_hidden=True,
                doc_boundaries=doc_boundaries,
            )
        else:
            h_rec = self.gru(
                x_rec,
                prev_hidden=prev_hidden,
                return_next_prev_hidden=False,
                doc_boundaries=doc_boundaries,
            )
            h_final_rec = None

        # Project up to model space
        output = self.project_up(h_rec)  # [B, T, H_rec] → [B, T, D]

        if return_next_prev_hidden:
            return output, h_final_rec
        else:
            return output

    def __repr__(self):
        return (
            f"ProjectedGRU(dim={self.dim}, h_recurrent={self.h_recurrent}, "
            f"compression={self.dim/self.h_recurrent:.1f}×)"
        )


class ProjectedGRUWithResidual(nn.Module):
    """
    ProjectedGRU with residual connection.

    Architecture:
        output = x + ProjectedGRU(x)

    This can help preserve capacity when H_rec << D.
    """

    def __init__(
        self,
        dim: int,
        h_recurrent: int = None,
        expansion_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.h_recurrent = h_recurrent or (dim // 2)

        self.projected_gru = ProjectedGRU(
            dim=dim,
            h_recurrent=self.h_recurrent,
            expansion_factor=expansion_factor,
            **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: torch.Tensor = None,
        return_next_prev_hidden: bool = False,
        doc_boundaries: torch.Tensor = None,
    ):
        """Forward with residual connection."""

        if return_next_prev_hidden:
            gru_out, h_final = self.projected_gru(
                x,
                prev_hidden=prev_hidden,
                return_next_prev_hidden=True,
                doc_boundaries=doc_boundaries,
            )
            output = x + gru_out  # Residual
            return output, h_final
        else:
            gru_out = self.projected_gru(
                x,
                prev_hidden=prev_hidden,
                return_next_prev_hidden=False,
                doc_boundaries=doc_boundaries,
            )
            output = x + gru_out  # Residual
            return output

    def __repr__(self):
        return (
            f"ProjectedGRUWithResidual(dim={self.dim}, h_recurrent={self.h_recurrent}, "
            f"compression={self.dim/self.h_recurrent:.1f}×)"
        )

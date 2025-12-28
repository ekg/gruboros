"""
LogSpaceWrapper - Universal log-space storage for any Elman cell

Wraps any cell (StockElman, GatedElman, etc.) with log-space hidden state storage.
This allows testing log-space benefit at ANY level of the ablation ladder.

Usage:
    from mingru.elman_ladder import StockElmanCell, LogSpaceWrapper

    cell = StockElmanCell(dim=512)
    log_cell = LogSpaceWrapper(cell)  # Now stores hidden state in log-space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_log_space(x, eps=1e-10):
    """Convert tensor to signed log representation: (log|x|, sign(x))"""
    sign_x = torch.sign(x)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    log_x = torch.log(torch.abs(x).clamp(min=eps))
    return log_x, sign_x


def from_log_space(log_x, sign_x):
    """Convert signed log representation back to linear."""
    return sign_x * torch.exp(log_x)


def log_rmsnorm(log_x, g, eps=1e-6):
    """RMSNorm in log-space without exponentiating.

    log(RMSNorm(exp(log_x))) = log_x - 0.5*logsumexp(2*log_x) + 0.5*log(n) + log(g)
    """
    n = log_x.shape[-1]
    log_mean_x_sq = torch.logsumexp(2 * log_x, dim=-1, keepdim=True) - torch.log(
        torch.tensor(n, dtype=log_x.dtype, device=log_x.device))
    log_rms = 0.5 * log_mean_x_sq
    log_normalized = log_x - log_rms
    log_g = torch.log(g.abs().clamp(min=eps))
    return log_normalized + log_g


class LogRMSNorm(nn.Module):
    """RMSNorm that operates in log-space."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, log_x):
        return log_rmsnorm(log_x, self.g, self.eps)


class RMSNorm(nn.Module):
    """Standard RMSNorm for bf16 stability."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.g


class LogSpaceCellWrapper(nn.Module):
    """
    Wraps any Elman cell with log-space hidden state storage.

    The wrapped cell computes h_new = cell(x, h_prev) normally.
    This wrapper:
    1. Converts log_h to h before calling cell
    2. Converts h to log_h after cell returns
    3. Uses LogRMSNorm for numerical stability
    """

    def __init__(self, cell, use_log_rmsnorm=True):
        super().__init__()
        self.cell = cell
        self.dim = cell.dim

        # Log-space normalization
        if use_log_rmsnorm:
            self.log_norm = LogRMSNorm(self.dim)
        else:
            self.log_norm = None

    def forward(self, x, log_h0=None, sign_h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            log_h0: [B, dim] initial log-magnitude of hidden state
            sign_h0: [B, dim] initial sign of hidden state

        Returns:
            log_h: [T+1, B, dim] log-magnitude of all hidden states
            sign_h: [T+1, B, dim] sign of all hidden states
        """
        T, B, D = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize log-space hidden state
        if log_h0 is None:
            log_h0 = torch.zeros(B, self.dim, device=device, dtype=torch.float32)
            sign_h0 = torch.ones(B, self.dim, device=device, dtype=torch.float32)

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]

        for t in range(T):
            # Convert from log-space to normal
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]

            # Apply LogRMSNorm before exponentiating (prevents overflow)
            if self.log_norm is not None:
                log_h_normed = self.log_norm(log_h_prev)
            else:
                log_h_normed = log_h_prev

            h_prev = from_log_space(log_h_normed, sign_h_prev).to(dtype)

            # Run single step of wrapped cell
            x_t = x[t:t+1]  # [1, B, dim]
            h_all = self.cell(x_t, h_prev)  # [2, B, dim] = [h_prev, h_new]
            h_new = h_all[-1]  # [B, dim]

            # Convert back to log-space
            log_h_new, sign_h_new = to_log_space(h_new)
            log_h_list.append(log_h_new.to(torch.float32))
            sign_h_list.append(sign_h_new.to(torch.float32))

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)

        return log_h, sign_h


class LogSpaceLayer(nn.Module):
    """
    Universal log-space layer for any Elman variant.

    Wraps any cell (StockElmanCell, GatedElmanCell, etc.) with:
    - Input/output projections
    - Log-space hidden state storage
    - LogRMSNorm before output
    - Residual connection

    This allows testing log-space benefit at ANY level of the ablation ladder.
    """

    def __init__(self, cell_class, dim, expansion=1.0, dropout=0.0,
                 use_log_rmsnorm=True, **cell_kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Create and wrap the cell with log-space storage
        cell = cell_class(self.d_inner, **cell_kwargs)
        self.log_cell = LogSpaceCellWrapper(cell, use_log_rmsnorm=use_log_rmsnorm)

        # RMSNorm before output projection (standard, not log-space)
        self.pre_out_norm = RMSNorm(self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, log_h0=None, sign_h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            log_h0: [B, d_inner] initial log-magnitude (optional)
            sign_h0: [B, d_inner] initial sign (optional)

        Returns:
            output: [B, T, dim] output sequence
            (log_h_final, sign_h_final): final hidden state in log-space
        """
        B, T, D = x.shape

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run log-space cell
        log_h_all, sign_h_all = self.log_cell(x_rnn, log_h0, sign_h0)

        # Get output hidden states (skip h0)
        log_h_out = log_h_all[1:]  # [T, B, d_inner]
        sign_h_out = sign_h_all[1:]

        # Final hidden state (in log-space)
        log_h_final = log_h_all[-1]
        sign_h_final = sign_h_all[-1]

        # Convert to normal space for output
        # Apply LogRMSNorm at the cell level, then exp
        h_out = from_log_space(log_h_out, sign_h_out)  # [T, B, d_inner]

        # Transpose back: [B, T, d_inner]
        h_out = h_out.permute(1, 0, 2).contiguous()

        # Apply standard RMSNorm, dropout, project
        h_out = self.pre_out_norm(h_out)
        h_out = self.dropout(h_out)
        output = self.out_proj(h_out)

        # Residual connection
        output = output + x

        return output, (log_h_final, sign_h_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LOG_SPACE=True'


# Convenience functions to create log-space variants of each level
def create_log_space_level(level, dim, expansion=1.0, dropout=0.0, **kwargs):
    """Create log-space variant of any ladder level.

    Args:
        level: 0-3 (Stock, Gated, Selective, DiagonalSelective)
        dim: Input/output dimension
        expansion: Hidden dimension multiplier
        dropout: Dropout rate
        **kwargs: Additional arguments for the cell

    Returns:
        LogSpaceLayer wrapping the specified cell type
    """
    from .stock_elman import StockElmanCell
    from .gated_elman import GatedElmanCell
    from .selective_elman import SelectiveElmanCell
    from .diagonal_selective import DiagonalSelectiveCell

    cell_classes = {
        0: StockElmanCell,
        1: GatedElmanCell,
        2: SelectiveElmanCell,
        3: DiagonalSelectiveCell,
    }

    if level not in cell_classes:
        raise ValueError(f"Level {level} not supported. Use 0-3.")

    return LogSpaceLayer(cell_classes[level], dim, expansion, dropout, **kwargs)


if __name__ == "__main__":
    print("Testing LogSpaceWrapper...")
    print("=" * 60)

    from stock_elman import StockElmanCell

    # Test wrapping StockElmanCell
    cell = StockElmanCell(dim=256)
    log_layer = LogSpaceLayer(StockElmanCell, dim=256, expansion=2.0)
    log_layer = log_layer.cuda().bfloat16()

    x = torch.randn(2, 32, 256, device='cuda', dtype=torch.bfloat16)

    print("Testing forward...")
    out, (log_h, sign_h) = log_layer(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Log hidden: {log_h.shape}, range [{log_h.min():.2f}, {log_h.max():.2f}]")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print("\n✓ LogSpaceWrapper test passed!")

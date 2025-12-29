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

    MEMORY-EFFICIENT: Does NOT store all T+1 hidden states.
    Only returns output sequence and final hidden state.
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
            h_out: [T, B, dim] output hidden states (in LINEAR space, not log)
            (log_h_final, sign_h_final): final hidden state in log-space
        """
        T, B, D = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize log-space hidden state (use input dtype, not float32!)
        if log_h0 is None:
            log_h0 = torch.zeros(B, self.dim, device=device, dtype=dtype)
            sign_h0 = torch.ones(B, self.dim, device=device, dtype=dtype)

        # Current hidden state in log-space
        log_h = log_h0
        sign_h = sign_h0

        # Collect outputs (in linear space) - NOT all hidden states
        h_out_list = []

        for t in range(T):
            # Apply LogRMSNorm before exponentiating (prevents overflow)
            if self.log_norm is not None:
                log_h_normed = self.log_norm(log_h)
            else:
                log_h_normed = log_h

            # Convert from log-space to linear for cell
            h_prev = from_log_space(log_h_normed, sign_h)

            # Run single step of wrapped cell
            x_t = x[t:t+1]  # [1, B, dim]
            h_all = self.cell(x_t, h_prev)  # [2, B, dim] = [h_prev, h_new]
            h_new = h_all[-1]  # [B, dim]

            # Store output (linear space)
            h_out_list.append(h_new)

            # Convert back to log-space for next iteration
            log_h, sign_h = to_log_space(h_new)

        # Stack outputs: [T, B, dim]
        h_out = torch.stack(h_out_list, dim=0)

        return h_out, (log_h, sign_h)


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

    def forward(self, x, prev_hidden=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            prev_hidden: Either None, or tuple (log_h0, sign_h0) from previous call

        Returns:
            output: [B, T, dim] output sequence
            (log_h_final, sign_h_final): final hidden state in log-space
        """
        B, T, D = x.shape

        # Unpack previous hidden state if provided
        if prev_hidden is None:
            log_h0, sign_h0 = None, None
        elif isinstance(prev_hidden, tuple) and len(prev_hidden) == 2:
            log_h0, sign_h0 = prev_hidden
        else:
            # Fallback: treat as log_h0, assume positive
            log_h0 = prev_hidden
            sign_h0 = None

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run log-space cell - now returns h_out directly (not log_h_all)
        h_out, (log_h_final, sign_h_final) = self.log_cell(x_rnn, log_h0, sign_h0)

        # h_out is already [T, B, d_inner] in linear space
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

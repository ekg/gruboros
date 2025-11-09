"""Standard GRU using PyTorch's nn.GRU (cuDNN backend)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class StandardGRU(nn.Module):
    """
    Wrapper around PyTorch's nn.GRU for compatibility with minLM.

    Uses cuDNN-optimized kernels (fastest available).
    This is the gold standard nonlinear GRU implementation.

    Supports gradient checkpointing to reduce memory usage.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        use_gradient_checkpointing=False,
        **kwargs  # Ignore other minGRU-specific args
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Standard GRU: hidden_size = dim_inner
        # Input projection: dim -> dim_inner
        self.input_proj = nn.Linear(dim, self.dim_inner, bias=False)

        # Core GRU cell (cuDNN backend)
        self.gru = nn.GRU(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,  # (batch, seq, feature)
            bias=True
        )

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

    def _gru_forward(self, x_proj, h0):
        """Helper function for gradient checkpointing."""
        return self.gru(x_proj, h0)

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True, return_next_prev_hidden=True, actual_length=None):
        """
        Args:
            x: (batch, seq_len, dim)
            prev_hiddens: Previous hidden state (batch, dim_inner) or None
            prev_conv_buffers: Unused (kept for API compatibility with conv layers)
            return_hiddens: Whether to return hidden states
            actual_length: Unused (kept for API compatibility)

        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, dim_inner) if return_hiddens else None
            next_conv_buffers: None (no conv in standard GRU)
        """
        batch, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (batch, seq_len, dim_inner)

        # Prepare initial hidden state
        if prev_hiddens is not None:
            # prev_hiddens is (batch, dim_inner), GRU expects (1, batch, dim_inner)
            h0 = prev_hiddens.unsqueeze(0)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # Run GRU with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointing: recompute forward during backward to save memory
            gru_out, hn = checkpoint(
                self._gru_forward,
                x_proj,
                h0,
                use_reentrant=False  # Use new non-reentrant checkpointing
            )
        else:
            gru_out, hn = self.gru(x_proj, h0)
        # gru_out: (batch, seq_len, dim_inner)
        # hn: (1, batch, dim_inner)

        # Project output
        output = self.output_proj(gru_out)  # (batch, seq_len, dim)

        if return_hiddens:
            # Return hidden state as (batch, dim_inner) and None for conv buffers
            next_hiddens = hn.squeeze(0)  # (batch, dim_inner)
            return output, next_hiddens, None  # (output, hiddens, conv_buffers)
        else:
            return output

    def __repr__(self):
        return f"StandardGRU(dim={self.dim}, dim_inner={self.dim_inner}, using cuDNN)"

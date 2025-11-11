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
        recurrence_chunk_size=64,  # Process sequence in chunks to save memory
        **kwargs  # Ignore other minGRU-specific args
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.recurrence_chunk_size = recurrence_chunk_size

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

    def forward(self, x, prev_hiddens=None, prev_conv_buffers=None, return_hiddens=True, return_next_prev_hidden=True, actual_length=None, doc_boundaries=None):
        """
        Args:
            x: (batch, seq_len, dim)
            prev_hiddens: Previous hidden state (batch, dim_inner) or None
            prev_conv_buffers: Unused (kept for API compatibility with conv layers)
            return_hiddens: Whether to return hidden states
            actual_length: Unused (kept for API compatibility)
            doc_boundaries: (batch, seq_len) bool tensor - True where document boundaries occur

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
            h0 = prev_hiddens.unsqueeze(0)  # (1, batch, dim_inner)
        else:
            h0 = torch.zeros(1, batch, self.dim_inner, device=x.device, dtype=x.dtype)

        # CHUNKED PROCESSING: Process in fixed recurrence_chunk_size chunks
        # Document boundaries handled OUTSIDE this function via hidden state resets
        # between chunks in the training loop
        chunk_size = self.recurrence_chunk_size

        gru_outputs = []
        h = h0

        # Process sequence in fixed-size chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Extract chunk
            x_chunk = x_proj[:, chunk_start:chunk_end]  # (batch, chunk_len, dim_inner)

            # Run cuDNN GRU on this chunk (optimized!)
            if self.use_gradient_checkpointing and self.training:
                chunk_out, h = checkpoint(self._gru_forward, x_chunk, h, use_reentrant=False)
            else:
                chunk_out, h = self.gru(x_chunk, h)

            gru_outputs.append(chunk_out)

        # Concatenate all chunk outputs
        gru_out = torch.cat(gru_outputs, dim=1)  # (batch, seq_len, dim_inner)
        hn = h  # Final hidden state

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

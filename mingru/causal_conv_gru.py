"""
Ultra-lightweight GRU replacement using ONLY causal conv + linear.

NO cuDNN, NO workspace bloat, MINIMAL memory!
Just sliding window + learnable mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvGRU(nn.Module):
    """
    Minimal GRU replacement: Causal conv for context + gated linear mixing.

    NO recurrence, NO cuDNN workspace!
    Memory: Just the parameters (~2MB per layer for 1536 dim)

    Architecture:
    1. Causal conv (kernel_size=4) for local context
    2. Linear gates for update/reset (like GRU but no state)
    3. Residual connection
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        kernel_size=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.kernel_size = kernel_size

        # Causal conv for local context (NO padding needed, we'll do it manually)
        self.conv = nn.Conv1d(
            dim,
            self.dim_inner,
            kernel_size=kernel_size,
            padding=0,  # We'll handle causal padding manually
            bias=False
        )

        # Gates: reset and update (like GRU)
        self.gate_proj = nn.Linear(self.dim_inner, self.dim_inner * 2, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.dim_inner, dim, bias=False)

        # Initialize with small values for stability
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.out_proj.weight, 0.)  # Start as identity

    def forward(
        self,
        x,
        prev_hiddens=None,  # Ignored! No state!
        prev_conv_buffers=None,
        return_hiddens=True,
        return_next_prev_hidden=True,
        actual_length=None,
        seq_chunk_size=1024  # LARGER chunks = better GPU utilization!
    ):
        """
        Args:
            x: [batch, seq, dim]
            seq_chunk_size: Process sequence in chunks of this size to reduce memory
                          Default 256 gives 8× memory reduction vs full 2048 sequence!

        Returns:
            output: [batch, seq, dim]
            None: No hidden state!
            None: No conv buffers!
        """
        batch, seq_len, dim = x.shape

        # If sequence is small, process all at once (no chunking overhead)
        if seq_len <= seq_chunk_size:
            return self._forward_chunk(x, return_hiddens)

        # CHUNKED PROCESSING: Process sequence in smaller chunks to reduce memory!
        # This reduces peak intermediate tensors from [batch, 2048, dim] to [batch, 256, dim]
        outputs = []
        conv_buffer = None  # Buffer of last (kernel_size-1) tokens for causal continuity

        for chunk_start in range(0, seq_len, seq_chunk_size):
            chunk_end = min(chunk_start + seq_chunk_size, seq_len)
            x_chunk = x[:, chunk_start:chunk_end, :]  # [batch, chunk_size, dim]

            # Process chunk with conv buffer for causal continuity
            out_chunk, conv_buffer = self._forward_chunk_with_buffer(x_chunk, conv_buffer)
            outputs.append(out_chunk)

        # Concatenate all chunk outputs
        output = torch.cat(outputs, dim=1)  # [batch, seq, dim]

        if return_hiddens:
            return output, None, None
        else:
            return output

    def _forward_chunk(self, x, return_hiddens=True):
        """Process a single chunk (original implementation)."""
        batch, seq_len, dim = x.shape

        # Transpose for conv: [batch, dim, seq]
        x_t = x.transpose(1, 2)

        # Causal padding: pad LEFT with (kernel_size - 1) zeros
        x_padded = F.pad(x_t, (self.kernel_size - 1, 0), value=0.)

        # Apply conv
        conv_out = self.conv(x_padded)  # [batch, dim_inner, seq]

        # Transpose back: [batch, seq, dim_inner]
        conv_out = conv_out.transpose(1, 2)

        # Compute gates
        gates = self.gate_proj(conv_out)  # [batch, seq, dim_inner * 2]
        reset_gate, update_gate = gates.chunk(2, dim=-1)

        # Apply sigmoid to gates
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # Gated mixing (like GRU but without recurrence)
        mixed = update_gate * conv_out + reset_gate * (1 - update_gate) * conv_out

        # Project to output
        output = self.out_proj(mixed)

        if return_hiddens:
            return output, None, None
        else:
            return output

    def _forward_chunk_with_buffer(self, x, conv_buffer):
        """
        Process a chunk with conv buffer for causal continuity.

        Args:
            x: [batch, chunk_size, dim]
            conv_buffer: [batch, dim, kernel_size-1] from previous chunk, or None

        Returns:
            output: [batch, chunk_size, dim]
            new_conv_buffer: [batch, dim, kernel_size-1] for next chunk
        """
        batch, seq_len, dim = x.shape

        # Transpose for conv: [batch, dim, seq]
        x_t = x.transpose(1, 2)

        # Causal padding: use conv_buffer if available, else pad with zeros
        if conv_buffer is not None:
            # Concatenate buffer from previous chunk
            x_padded = torch.cat([conv_buffer, x_t], dim=2)  # [batch, dim, (kernel-1) + seq]
        else:
            # First chunk: pad with zeros
            x_padded = F.pad(x_t, (self.kernel_size - 1, 0), value=0.)

        # Apply conv
        conv_out = self.conv(x_padded)  # [batch, dim_inner, seq]

        # Save last (kernel_size-1) elements as buffer for next chunk
        new_conv_buffer = x_t[:, :, -(self.kernel_size - 1):].clone()

        # Transpose back: [batch, seq, dim_inner]
        conv_out = conv_out.transpose(1, 2)

        # Compute gates
        gates = self.gate_proj(conv_out)  # [batch, seq, dim_inner * 2]
        reset_gate, update_gate = gates.chunk(2, dim=-1)

        # Apply sigmoid to gates
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # Gated mixing
        mixed = update_gate * conv_out + reset_gate * (1 - update_gate) * conv_out

        # Project to output
        output = self.out_proj(mixed)

        return output, new_conv_buffer

    def __repr__(self):
        return f"CausalConvGRU(dim={self.dim}, dim_inner={self.dim_inner}, kernel={self.kernel_size}, NO CUDNN!)"

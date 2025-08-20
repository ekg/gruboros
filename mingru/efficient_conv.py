"""Efficient causal convolution with persistent buffers for streaming."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCausalConv1d(nn.Module):
    """
    Efficient causal convolution that maintains a persistent buffer
    to avoid concatenation overhead during streaming.
    """
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding_size = kernel_size - 1
        
        # Convolution layers
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                   groups=dim, bias=False, padding=0)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        
        # Register a persistent buffer for the padding
        # This avoids reallocation and allows for in-place updates
        self.register_buffer('padding_buffer', None)
        
    def forward(self, x, use_buffer=True, reset_buffer=False):
        """
        Forward pass with efficient buffering.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            use_buffer: Whether to use the persistent buffer (for training)
            reset_buffer: Whether to reset the buffer (at document boundaries)
        
        Returns:
            output: Convolved output [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Transpose for conv1d: [batch, seq_len, dim] -> [batch, dim, seq_len]
        x = x.transpose(1, 2)
        
        if use_buffer and self.training:
            # Initialize buffer if needed
            if self.padding_buffer is None or reset_buffer:
                self.padding_buffer = torch.zeros(
                    batch_size, dim, self.padding_size, 
                    dtype=x.dtype, device=x.device
                )
            
            # Efficient approach: Use F.conv1d directly with the buffer
            # Concatenate buffer with input for convolution
            # Note: We use torch.cat here but only once per forward, not in a loop
            padded_x = torch.cat([self.padding_buffer, x], dim=2)
            
            # Update buffer IN-PLACE for next iteration (no new allocation)
            # Use .data to avoid autograd tracking
            if seq_len >= self.padding_size:
                self.padding_buffer.data.copy_(x[:, :, -self.padding_size:])
            else:
                # Handle short sequences
                self.padding_buffer.data[:, :, :-seq_len].copy_(
                    self.padding_buffer.data[:, :, seq_len:]
                )
                self.padding_buffer.data[:, :, -seq_len:].copy_(x)
        else:
            # Inference or when buffer is disabled: use standard zero padding
            padded_x = F.pad(x, (self.padding_size, 0), value=0.)
        
        # Apply convolutions
        x = self.depthwise(padded_x)
        x = self.pointwise(x)
        
        # Transpose back: [batch, dim, seq_len] -> [batch, seq_len, dim]
        x = x.transpose(1, 2)
        
        return x
    
    def reset_buffer(self):
        """Reset the padding buffer (e.g., at document boundaries)."""
        if self.padding_buffer is not None:
            self.padding_buffer.zero_()


class OptimizedCausalDepthWiseConv1d(nn.Module):
    """
    Optimized version using unfold for better memory access patterns.
    """
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding_size = kernel_size - 1
        
        # Single fused conv operation
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=0),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        
        # Persistent buffer
        self.register_buffer('buffer', None)
        
    def forward(self, x, prev_buffer=None):
        """
        Optimized forward with minimal memory operations.
        """
        batch_size, seq_len, dim = x.shape
        
        # Transpose once
        x_conv = x.transpose(1, 2)  # [B, D, L]
        
        # Handle buffering efficiently
        if prev_buffer is not None:
            # prev_buffer is already [B, D, padding_size]
            x_padded = torch.cat([prev_buffer, x_conv], dim=2)
        else:
            # First chunk - use zero padding
            x_padded = F.pad(x_conv, (self.padding_size, 0), value=0.)
        
        # Extract next buffer (no detach needed if we use .data later)
        if seq_len >= self.padding_size:
            next_buffer = x_conv[:, :, -self.padding_size:].contiguous()
        else:
            next_buffer = x_padded[:, :, -self.padding_size:].contiguous()
        
        # Apply convolution
        output = self.net(x_padded)
        
        # Transpose back
        output = output.transpose(1, 2)  # [B, L, D]
        
        return output, next_buffer
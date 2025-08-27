"""
FixedGRU that matches NAU_GRU interface for drop-in replacement in training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NAU_GRU(nn.Module):
    """
    Fixed NAU_GRU with proper GRU mathematics.
    
    This is a drop-in replacement for the buggy NAU_GRU that uses
    proper GRU gates instead of minGRU's log-space activation.
    
    Maintains the same interface as the original for compatibility.
    """
    
    def __init__(
        self, 
        dim: int, 
        expansion_factor: float = 1.5,
        use_nau: bool = True,  # Kept for compatibility
        use_barriers: bool = False,  # Simplified - barriers not needed for proper GRU
        barrier_min: float = -8.0,
        barrier_max: float = 8.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_barriers = use_barriers  # Keep for interface compatibility
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
        
        # Combined projections for efficiency (matches original interface)
        # Input projections for all 3 GRU gates
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=True)
        
        # Hidden projections for all 3 GRU gates  
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner, bias=True)
        
        # Output projection
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize weights properly
        self._init_weights()
        
        print(f"FixedGRU initialized: dim={dim}, dim_inner={self.dim_inner}, expansion={expansion_factor:.2f}")
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        import math
        
        # Standard GRU initialization (matches PyTorch)
        std = 1.0 / math.sqrt(self.dim_inner)
        
        # Input and hidden projections
        nn.init.uniform_(self.input_projection.weight, -std, std)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.uniform_(self.hidden_projection.weight, -std, std)
        nn.init.zeros_(self.hidden_projection.bias)
        
        # Output projection - small for residual connections
        nn.init.normal_(self.to_out.weight, mean=0.0, std=0.02)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        """
        Forward pass with proper GRU dynamics + residual connection.
        
        Maintains the same interface as original NAU_GRU.
        """
        # Handle input dimension variations
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, dim], got {x.shape}")
            
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        if input_dim != self.dim:
            raise ValueError(f"Input dim {input_dim} != expected {self.dim}")
        
        # Initialize hidden state (NO extreme values like -20.0!)
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            # Handle various hidden state formats
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            
            if prev_hidden.shape[0] != batch_size:
                # Batch size changed - use zeros instead of extreme values
                h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
            else:
                # For FixedGRU, hidden state is in regular space (not log space)
                h = prev_hidden
                
                # If it's in log space from old NAU_GRU, convert it
                if h.abs().mean() > 10:  # Likely log space values
                    h = torch.clamp(h, min=-10, max=10)
                    h = torch.exp(h)
                    print("Converted log-space hidden state to regular space")
        
        outputs = []
        
        # Process sequence timestep by timestep (required for RNN dynamics)
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, input_dim]
            
            # Project input and hidden state
            # Stay in original dtype - NO precision-losing conversions!
            input_gates = self.input_projection(x_t)  # [batch, 3*hidden]
            hidden_gates = self.hidden_projection(h)  # [batch, 3*hidden]
            
            # Split into the 3 GRU gates
            i_r, i_z, i_n = input_gates.chunk(3, dim=1)
            h_r, h_z, h_n = hidden_gates.chunk(3, dim=1)
            
            # Proper GRU equations:
            
            # 1. Reset gate - controls how much past info to forget
            reset_gate = torch.sigmoid(i_r + h_r)
            
            # 2. Update gate - controls how much to update vs retain
            update_gate = torch.sigmoid(i_z + h_z)
            
            # 3. New gate (candidate) - this is the KEY non-linearity!
            # This makes RNNs non-associative and expressive
            new_gate = torch.tanh(i_n + reset_gate * h_n)
            
            # 4. GRU update rule - interpolate between old and new
            h = (1.0 - update_gate) * h + update_gate * new_gate
            
            # Optional barriers (for compatibility with original interface)
            if self.use_barriers:
                h = torch.clamp(h, min=-10.0, max=10.0)  # Reasonable limits
            
            outputs.append(h)
        
        # Stack outputs efficiently
        h_sequence = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
        
        # Project to output dimension
        gru_output = self.to_out(h_sequence)  # [batch, seq, input_dim]
        
        # CRITICAL: Add residual connection for gradient flow!
        # This matches minLM's pattern and fixes the gradient flow issues
        output = gru_output + x
        
        # Return in the same format as original NAU_GRU
        if not return_next_prev_hidden:
            return output
        else:
            # Return hidden state in regular space (not log space)
            return output, h


# Alias for compatibility
FixedNAUGRU = NAU_GRU


if __name__ == "__main__":
    # Test the implementation
    print("Testing FixedGRU with NAU_GRU interface...")
    
    batch_size = 2
    seq_len = 16
    dim = 64
    
    model = NAU_GRU(dim=dim, expansion_factor=1.5)
    x = torch.randn(batch_size, seq_len, dim) * 0.1
    
    # Test forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    
    # Test with hidden state
    output2, final_h = model(x, return_next_prev_hidden=True)
    print(f"Final hidden shape: {final_h.shape}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("✓ Gradients computed successfully")
    
    # Test numerical stability
    x_large = torch.randn(2, 8, dim) * 10
    output_large = model(x_large)
    if torch.isfinite(output_large).all():
        print("✓ Numerically stable with large inputs")
    else:
        print("✗ Numerical instability detected")
        
    print("FixedGRU ready for training!")
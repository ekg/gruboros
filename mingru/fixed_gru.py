"""
Fixed GRU implementation that addresses:
1. Proper GRU mathematics 
2. Precision issues with dtype conversions
3. Gradient flow problems (residual connections, exp operations)
4. Proper initialization for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FixedGRU(nn.Module):
    """
    Fixed GRU that addresses all the NAU_GRU problems:
    
    1. Uses proper GRU equations (not minGRU log-space activation)
    2. Avoids dtype conversion precision issues  
    3. Includes residual connection for gradient flow
    4. Better initialization
    5. No problematic exp() operations in forward path
    """
    
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Combined linear layers for efficiency (like original NAU_GRU)
        # But with proper GRU gates
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner, bias=True)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner, bias=True)
        
        # Output projection
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization that avoids gradient flow problems"""
        import math
        
        # Standard GRU initialization
        std = 1.0 / math.sqrt(self.dim_inner)
        
        # Input projection (Xavier uniform like PyTorch GRU)
        nn.init.uniform_(self.input_projection.weight, -std, std)
        nn.init.zeros_(self.input_projection.bias)
        
        # Hidden projection  
        nn.init.uniform_(self.hidden_projection.weight, -std, std)
        nn.init.zeros_(self.hidden_projection.bias)
        
        # Initialize output projection small for residual connections
        # This is CRITICAL for gradient flow
        nn.init.normal_(self.to_out.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_hidden: Optional[torch.Tensor] = None,
        return_next_prev_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with proper GRU and residual connections
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state (NO extreme values like -20.0!)
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            # Handle dimension changes gracefully
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            if prev_hidden.shape[0] != batch_size:
                # Don't reset to extreme values - use zero
                h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
            else:
                h = prev_hidden
        
        outputs = []
        
        # Process sequence (required for RNN dynamics)
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, input_dim]
            
            # Project input and hidden state
            # Stay in original dtype - no precision-losing conversions!
            input_gates = self.input_projection(x_t)  # [batch, 3*hidden]
            hidden_gates = self.hidden_projection(h)  # [batch, 3*hidden]
            
            # Split into gates
            i_r, i_z, i_n = input_gates.chunk(3, dim=1)
            h_r, h_z, h_n = hidden_gates.chunk(3, dim=1)
            
            # Reset gate
            reset_gate = torch.sigmoid(i_r + h_r)
            
            # Update gate  
            update_gate = torch.sigmoid(i_z + h_z)
            
            # New gate (candidate) - this is the key non-linearity!
            new_gate = torch.tanh(i_n + reset_gate * h_n)
            
            # GRU update - standard equation
            h = (1.0 - update_gate) * h + update_gate * new_gate
            
            outputs.append(h)
        
        # Stack outputs
        h_sequence = torch.stack(outputs, dim=1)
        
        # Project to output dimension
        gru_output = self.to_out(h_sequence)
        
        # CRITICAL: Add residual connection for gradient flow!
        # This matches minLM's pattern: x = layer(x) + x
        output = gru_output + x
        
        if return_next_prev_hidden:
            return output, h
        return output


def test_against_builtin():
    """Test our implementation structure (not exact numerical match)"""
    print("Testing FixedGRU structure and gradient flow...")
    
    batch_size = 2
    seq_len = 8
    dim = 32
    
    # Create model and data
    model = FixedGRU(dim=dim, expansion_factor=1.5)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    # Forward pass
    output = model(x)
    
    # Check shapes
    assert output.shape == x.shape, f"Output {output.shape} != input {x.shape}"
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: {name} has no gradient!")
        else:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
    
    # Test with previous hidden state
    h0 = torch.randn(batch_size, model.dim_inner)
    output2, final_h = model(x, prev_hidden=h0, return_next_prev_hidden=True)
    assert final_h.shape == (batch_size, model.dim_inner)
    
    print("✓ FixedGRU passes structure tests!")
    
    # Test numerical stability
    # Create extreme inputs to test stability
    x_extreme = torch.randn(2, 4, dim) * 100  # Large inputs
    try:
        output_extreme = model(x_extreme)
        if torch.isfinite(output_extreme).all():
            print("✓ Model is numerically stable with large inputs")
        else:
            print("✗ Model produces NaN/Inf with large inputs")
    except Exception as e:
        print(f"✗ Model fails with large inputs: {e}")


if __name__ == "__main__":
    test_against_builtin()
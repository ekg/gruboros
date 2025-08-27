"""
Proper GRU implementation with correct mathematical formulation.

This fixes the NAU_GRU bug where it was using minGRU's log-space activation
instead of proper GRU gates (reset, update, candidate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProperGRU(nn.Module):
    """
    Proper GRU implementation following standard equations:
    
    r_t = σ(W_xr @ x_t + W_hr @ h_{t-1})    # Reset gate
    z_t = σ(W_xz @ x_t + W_hz @ h_{t-1})    # Update gate  
    n_t = tanh(W_xn @ x_t + W_hn @ (r_t ⊙ h_{t-1}))  # Candidate
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t  # Update
    
    This is the non-associative computation that makes RNNs expressive.
    """
    
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Input-to-hidden weights (3 gates)
        self.W_xr = nn.Linear(dim, self.dim_inner, bias=False)  # Reset gate
        self.W_xz = nn.Linear(dim, self.dim_inner, bias=False)  # Update gate
        self.W_xn = nn.Linear(dim, self.dim_inner, bias=False)  # Candidate
        
        # Hidden-to-hidden weights (3 gates)
        self.W_hr = nn.Linear(self.dim_inner, self.dim_inner, bias=False)  # Reset gate
        self.W_hz = nn.Linear(self.dim_inner, self.dim_inner, bias=False)  # Update gate
        self.W_hn = nn.Linear(self.dim_inner, self.dim_inner, bias=False)  # Candidate
        
        # Output projection
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following PyTorch's GRU initialization"""
        import math
        
        # Standard deviation for weight initialization
        std = 1.0 / math.sqrt(self.dim_inner)
        
        # Initialize input-to-hidden weights
        for layer in [self.W_xr, self.W_xz, self.W_xn]:
            nn.init.uniform_(layer.weight, -std, std)
            
        # Initialize hidden-to-hidden weights  
        for layer in [self.W_hr, self.W_hz, self.W_hn]:
            nn.init.uniform_(layer.weight, -std, std)
            
        # Initialize output layer small for residual connections
        nn.init.constant_(self.to_out.weight, 0.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_hidden: Optional[torch.Tensor] = None,
        return_next_prev_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass implementing proper GRU dynamics.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            prev_hidden: Previous hidden state [batch_size, hidden_dim]
            return_next_prev_hidden: Whether to return final hidden state
            
        Returns:
            output: Output tensor [batch_size, seq_len, input_dim]
            final_hidden: Final hidden state (if return_next_prev_hidden=True)
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state if not provided
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            # Handle dimension mismatch (e.g. from minGRU format)
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            if prev_hidden.shape[0] != batch_size:
                h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
            else:
                h = prev_hidden
        
        outputs = []
        
        # Process sequence timestep by timestep (required for proper GRU)
        for t in range(seq_len):
            x_t = x[:, t]  # Current input [batch_size, input_dim]
            
            # Compute gates
            # Reset gate: controls how much past info to forget
            r_t = torch.sigmoid(self.W_xr(x_t) + self.W_hr(h))
            
            # Update gate: controls how much to update vs retain  
            z_t = torch.sigmoid(self.W_xz(x_t) + self.W_hz(h))
            
            # Candidate: new content with reset-gated hidden state
            # This is the key non-linearity that makes GRU expressive!
            n_t = torch.tanh(self.W_xn(x_t) + self.W_hn(r_t * h))
            
            # Update hidden state: interpolate between old and new
            h = (1 - z_t) * h + z_t * n_t
            
            outputs.append(h)
        
        # Stack outputs
        h_sequence = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
        
        # Project to output dimension
        output = self.to_out(h_sequence)  # [batch, seq, input_dim]
        
        if return_next_prev_hidden:
            return output, h
        return output


def compare_with_pytorch_gru():
    """
    Compare our implementation with PyTorch's built-in GRU.
    This is crucial for verifying correctness.
    """
    print("Comparing ProperGRU with PyTorch's nn.GRU...")
    
    # Test parameters
    batch_size = 4
    seq_len = 16
    input_dim = 64
    hidden_dim = 96  # expansion_factor = 1.5
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_dim)
    h0 = torch.randn(batch_size, hidden_dim)
    
    # Our implementation
    our_gru = ProperGRU(dim=input_dim, expansion_factor=1.5)
    
    # PyTorch's implementation (single layer, no bias to match ours)
    pytorch_gru = nn.GRU(
        input_size=input_dim, 
        hidden_size=hidden_dim,
        num_layers=1,
        bias=False,
        batch_first=True
    )
    
    # Add output projection to PyTorch GRU to match our interface
    pytorch_output_proj = nn.Linear(hidden_dim, input_dim, bias=False)
    
    # Copy weights to make them identical
    with torch.no_grad():
        # PyTorch GRU has weight_ih_l0 and weight_hh_l0
        # weight_ih_l0 is [3*hidden, input] stacked as [reset, update, new]
        # weight_hh_l0 is [3*hidden, hidden] stacked as [reset, update, new]
        
        ih_weight = pytorch_gru.weight_ih_l0  # [3*hidden, input]
        hh_weight = pytorch_gru.weight_hh_l0  # [3*hidden, hidden]
        
        # Split into gate weights
        ih_r, ih_z, ih_n = ih_weight.chunk(3, dim=0)
        hh_r, hh_z, hh_n = hh_weight.chunk(3, dim=0)
        
        # Copy to our model
        our_gru.W_xr.weight.copy_(ih_r)
        our_gru.W_xz.weight.copy_(ih_z)  
        our_gru.W_xn.weight.copy_(ih_n)
        our_gru.W_hr.weight.copy_(hh_r)
        our_gru.W_hz.weight.copy_(hh_z)
        our_gru.W_hn.weight.copy_(hh_n)
        
        # Initialize output projections to be identical  
        our_gru.to_out.weight.copy_(pytorch_output_proj.weight)
    
    # Forward pass
    with torch.no_grad():
        # PyTorch GRU expects hidden as [num_layers, batch, hidden]
        h0_pytorch = h0.unsqueeze(0)  # [1, batch, hidden]
        pytorch_hidden_seq, pytorch_final = pytorch_gru(x, h0_pytorch)
        pytorch_final = pytorch_final.squeeze(0)  # Remove layer dim
        pytorch_output = pytorch_output_proj(pytorch_hidden_seq)  # Project to input_dim
        
        # Our GRU
        our_output, our_final = our_gru(x, h0, return_next_prev_hidden=True)
    
    # Compare outputs
    output_diff = torch.max(torch.abs(pytorch_output - our_output))
    final_diff = torch.max(torch.abs(pytorch_final - our_final))
    
    print(f"Max output difference: {output_diff:.6f}")
    print(f"Max final hidden difference: {final_diff:.6f}")
    
    if output_diff < 1e-5 and final_diff < 1e-5:
        print("✓ Our implementation matches PyTorch's GRU!")
        return True
    else:
        print("✗ Implementation differs from PyTorch's GRU")
        return False


if __name__ == "__main__":
    compare_with_pytorch_gru()
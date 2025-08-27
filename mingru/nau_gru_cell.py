"""NAU-GRU using torch RNN cell pattern for efficient sequential processing"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit, Tensor
from typing import Tuple


@jit.script
def gru_cell(h_log: Tensor, h_t: Tensor, g_t: Tensor, use_barriers: bool, barrier_min: float, barrier_max: float) -> Tensor:
    """JIT-compiled GRU cell operation - h_log is already in log space"""
    # Activation with numerical stability
    h_new = torch.where(
        h_t >= 0,
        torch.log(F.relu(h_t) + 1e-8),  # Remove +0.5 for sparsity
        -F.softplus(-h_t)
    )
    
    # Gate log probabilities
    log_gate = -F.softplus(-g_t)            # log(sigmoid(g_t))
    log_one_minus_gate = -F.softplus(g_t)   # log(1 - sigmoid(g_t))
    
    # Log-sum-exp (h_log is ALREADY in log space)
    h_log_new = torch.logaddexp(
        log_one_minus_gate + h_log,
        log_gate + h_new
    )
    
    # Apply energy barriers
    if use_barriers:
        # Soft barrier forces
        lower_force = 0.5 * torch.exp(barrier_min - h_log_new + 5.0)
        upper_force = 0.5 * torch.exp(h_log_new - barrier_max + 5.0)
        h_log_new = h_log_new + lower_force - upper_force
        # Hard clamp
        h_log_new = torch.clamp(h_log_new, min=barrier_min, max=barrier_max)
    else:
        h_log_new = torch.clamp(h_log_new, min=-20.0, max=20.0)
    
    return h_log_new  # Return log space, not exp!


class NAU_GRU(nn.Module):
    """Efficient sequential GRU using custom cell"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, use_barriers: bool = True, barrier_min: float = -10.0, barrier_max: float = 10.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_barriers = use_barriers
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
        
        # Combined projection for efficiency
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Match minLM's careful initialization
        import math
        std = 0.02 / math.sqrt(dim)
        nn.init.normal_(self.to_hidden_and_gate.weight, mean=0.0, std=std)
        # CRITICAL: Zero-initialize output for residual identity
        nn.init.constant_(self.to_out.weight, 0.0)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        # Get dimensions
        if x.dim() == 4 and x.size(2) == 1:
            # Squeeze out singleton dimension if present
            x = x.squeeze(2)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, dim], got shape {x.shape}")
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Sanity check input
        if input_dim != self.dim:
            raise ValueError(f"Input dim {input_dim} != expected dim {self.dim}")
        
        # Single projection for all timesteps
        combined = self.to_hidden_and_gate(x)
        if combined.shape != (batch_size, seq_len, self.dim_inner * 2):
            raise ValueError(f"combined shape {combined.shape} != expected {(batch_size, seq_len, self.dim_inner * 2)}")
        
        hidden_seq, gate_seq = combined.chunk(2, dim=-1)
        if hidden_seq.shape != (batch_size, seq_len, self.dim_inner):
            raise ValueError(f"hidden_seq shape {hidden_seq.shape} != expected {(batch_size, seq_len, self.dim_inner)}")
        
        # Initialize state IN LOG SPACE
        if prev_hidden is None:
            # Start at -5.0: exp(-5) ≈ 0.0067 - allows learning sparse representations
            h_log = torch.full((batch_size, self.dim_inner), -5.0, device=device, dtype=dtype)
        else:
            if prev_hidden.shape != (batch_size, self.dim_inner):
                # Batch size changed - reinitialize
                h_log = torch.full((batch_size, self.dim_inner), -20.0, device=device, dtype=dtype)
            else:
                h_log = prev_hidden  # Already in log space
        
        # Process sequence
        output_list = []
        
        # Use the JIT-compiled cell
        for t in range(seq_len):
            h_t = hidden_seq[:, t]  # Should be [batch_size, dim_inner]
            g_t = gate_seq[:, t]    # Should be [batch_size, dim_inner]
            
            # Debug check
            if h_t.shape != (batch_size, self.dim_inner):
                raise ValueError(f"h_t shape {h_t.shape} != expected {(batch_size, self.dim_inner)}")
            
            h_log = gru_cell(h_log, h_t, g_t, self.use_barriers, self.barrier_min, self.barrier_max)
            
            # Check output shape
            if h_log.shape != (batch_size, self.dim_inner):
                raise ValueError(f"gru_cell output shape {h_log.shape} != expected {(batch_size, self.dim_inner)}")
                
            output_list.append(torch.exp(h_log))  # Only exp for output
        
        # Stack outputs efficiently
        h_outputs = torch.stack(output_list, dim=1)
        
        # Ensure we have exactly seq_len outputs
        assert h_outputs.size(1) == seq_len, f"Output seq_len {h_outputs.size(1)} != input seq_len {seq_len}"
        assert h_outputs.shape == (batch_size, seq_len, self.dim_inner), f"h_outputs shape {h_outputs.shape} != expected {(batch_size, seq_len, self.dim_inner)}"
        
        # Project to output dimension
        output = self.to_out(h_outputs)
        
        # Final sanity check
        if output.shape != (batch_size, seq_len, self.dim):
            print(f"Debug: input x.shape={x.shape}, batch_size={batch_size}, seq_len={seq_len}, self.dim={self.dim}")
            print(f"Debug: h_outputs.shape={h_outputs.shape}, output.shape={output.shape}")
            assert False, f"Output shape {output.shape} != expected {(batch_size, seq_len, self.dim)}"
        
        # Return
        if not return_next_prev_hidden:
            return output
        return output, h_log  # Return log-space hidden state
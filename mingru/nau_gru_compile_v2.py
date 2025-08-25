"""Simplified compile-compatible NAU-GRU"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAU_GRU(nn.Module):
    """Extremely simplified NAU-GRU for debugging compilation"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, 
                 use_nau: bool = True, use_barriers: bool = True,
                 barrier_min: float = -10, barrier_max: float = 10):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Standard minGRU components
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if prev_hidden is None:
            prev_hidden = x.new_zeros(batch_size, self.dim_inner)
        
        outputs = []
        hidden = prev_hidden
        
        # Process sequence step by step (simple version)
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Compute gates
            h_and_g = self.to_hidden_and_gate(x_t)
            hidden_pre, gate = h_and_g.chunk(2, dim=-1)
            
            # Apply activation
            hidden_new = torch.where(
                hidden_pre >= 0,
                (F.relu(hidden_pre) + 0.5).log(),
                -F.softplus(-hidden_pre)
            )
            
            # Mix with previous hidden
            gate_sigmoid = torch.sigmoid(gate)
            log_hidden = torch.log(hidden.abs() + 1e-8)
            log_hidden = (1 - gate_sigmoid) * log_hidden + gate_sigmoid * hidden_new
            hidden = torch.exp(log_hidden)
            
            # Output projection
            out = self.to_out(hidden)
            outputs.append(out)
        
        output = torch.stack(outputs, dim=1)
        
        if not return_next_prev_hidden:
            return output
        return output, hidden
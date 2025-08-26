"""NAU_GRU using torch.jit.script for accelerated sequential processing"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.jit.script
def nau_sequential_scan(hidden: torch.Tensor, gate: torch.Tensor, 
                        prev_hidden: torch.Tensor,
                        W_add: torch.Tensor, W_mul: torch.Tensor) -> torch.Tensor:
    """JIT-compiled sequential scan for NAU operations"""
    batch_size, seq_len, dim = hidden.shape
    outputs = []
    h = prev_hidden
    
    for t in range(seq_len):
        h_t = hidden[:, t]
        g_t = gate[:, t]
        
        # NAU arithmetic operations
        # Additive path
        add_out = torch.matmul(h_t, W_add)
        # Multiplicative path (in log space)
        mul_out = torch.exp(torch.matmul(torch.log(torch.abs(h_t) + 1e-8), W_mul))
        
        # Gate between operations
        gate_mix = torch.sigmoid(g_t)
        h_new = gate_mix * add_out + (1 - gate_mix) * mul_out
        
        # Update hidden state
        h = h_new
        outputs.append(h)
    
    return torch.stack(outputs, dim=1)


class NAU_GRU(nn.Module):
    """NAU-GRU with torch.jit acceleration for sequential processing"""
    def __init__(self, dim: int, expansion_factor: float = 1.5,
                 use_nau: bool = True, use_barriers: bool = True,
                 barrier_min: float = -10, barrier_max: float = 10):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_nau = use_nau
        
        # Standard minGRU components
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # NAU weight matrices
        if use_nau:
            self.W_add = nn.Parameter(torch.randn(self.dim_inner, self.dim_inner) * 0.02)
            self.W_mul = nn.Parameter(torch.randn(self.dim_inner, self.dim_inner) * 0.02)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden
        if prev_hidden is None:
            prev_hidden = x.new_zeros(batch_size, self.dim_inner)
        
        # Process input
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        
        if self.use_nau and hasattr(self, 'W_add'):
            # Use JIT-compiled scan
            h_seq = nau_sequential_scan(hidden, gate, prev_hidden, self.W_add, self.W_mul)
        else:
            # Fallback to simple parallel processing
            # Use associative scan like minGRU
            log_gate = -F.softplus(gate)  # log(1 - sigmoid)
            log_hidden = torch.where(
                hidden >= 0,
                (F.relu(hidden) + 0.5).log(),
                -F.softplus(-hidden)
            )
            
            # Simple parallel scan approximation
            if seq_len == 1:
                gate_sigmoid = torch.sigmoid(gate[:, 0])
                h_new = torch.exp(log_hidden[:, 0])
                h_seq = ((1 - gate_sigmoid) * prev_hidden + gate_sigmoid * h_new).unsqueeze(1)
            else:
                # Cumulative product for gates
                log_gate_cumprod = torch.cumsum(log_gate, dim=1)
                # Apply gates
                h_seq = torch.exp(log_hidden + log_gate_cumprod)
        
        # Output projection
        output = self.to_out(h_seq)
        
        # Get final hidden state
        next_hidden = h_seq[:, -1]
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
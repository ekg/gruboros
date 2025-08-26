"""NAU-GRU with forced sequential processing that maintains exact shape compatibility"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NAU_GRU(nn.Module):
    """NAU-GRU that forces sequential processing while maintaining minGRU interface"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, 
                 use_nau: bool = True, use_barriers: bool = True,
                 barrier_min: float = -10, barrier_max: float = 10):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_nau = use_nau
        self.use_barriers = use_barriers
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
        
        # Standard minGRU components
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # NAU components (simplified)
        if use_nau:
            self.W_gate = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        
        # Barrier strength
        if use_barriers:
            self.barrier_strength = nn.Parameter(torch.ones(self.dim_inner) * 0.1)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        if use_nau:
            nn.init.xavier_uniform_(self.W_gate.weight)
            nn.init.zeros_(self.W_gate.bias)
    
    @torch.jit.ignore
    def forward_sequential(self, x, prev_hidden=None):
        """Force sequential processing by disabling JIT optimization"""
        batch_size, seq_len, _ = x.shape
        
        # Process all at once for efficiency
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        
        # Initialize hidden state
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=x.device, dtype=x.dtype)
        else:
            h = prev_hidden
            
        outputs = []
        
        # Force sequential processing
        for t in range(seq_len):
            h_t = hidden[:, t]  # [batch, dim_inner]
            g_t = gate[:, t]    # [batch, dim_inner]
            
            # Log-space activation
            if self.use_nau and hasattr(self, 'W_gate'):
                # Simple gated arithmetic operation
                gate_add = torch.sigmoid(self.W_gate(h_t))
                h_log = torch.log(torch.abs(h) + 1e-8)
                h_t_log = torch.log(torch.abs(h_t) + 1e-8)
                h_new = gate_add * torch.exp(h_log + h_t_log) + (1 - gate_add) * (h + h_t)
                h_new = torch.log(torch.abs(h_new) + 1e-8)
            else:
                # Standard minGRU log activation
                h_new = torch.where(
                    h_t >= 0,
                    (F.relu(h_t) + 0.5).log(),
                    -F.softplus(-h_t)
                )
            
            # Apply barriers in log space
            if self.use_barriers:
                h_log = torch.log(torch.abs(h) + 1e-8) if not self.use_nau else h_new
                lower_force = -self.barrier_strength / (h_log - self.barrier_min + 0.1)
                upper_force = self.barrier_strength / (self.barrier_max - h_log + 0.1)
                h_log = h_log + lower_force + upper_force
                h_log = torch.clamp(h_log, self.barrier_min + 0.1, self.barrier_max - 0.1)
                h = torch.exp(h_log)
            else:
                h_exp = torch.exp(h_new) if not (self.use_nau and hasattr(self, 'W_gate')) else h_new
                h = h_exp
            
            # Apply gate
            g_sigmoid = torch.sigmoid(g_t)
            h = (1 - g_sigmoid) * h + g_sigmoid * torch.exp(h_new) if not (self.use_nau and self.use_barriers) else (1 - g_sigmoid) * h + g_sigmoid * h
            
            # Output projection
            out_t = self.to_out(h)
            outputs.append(out_t)
        
        # Stack outputs - this ensures correct shape [batch, seq_len, dim]
        output = torch.stack(outputs, dim=1)
        
        return output, h
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        """Main forward - force sequential processing"""
        output, next_hidden = self.forward_sequential(x, prev_hidden)
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
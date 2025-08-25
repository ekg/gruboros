"""Compile-compatible NAU-GRU that matches minGRU interface exactly"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NAU_GRU(nn.Module):
    """Simplified NAU-GRU that compiles cleanly"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, 
                 use_nau: bool = True, use_barriers: bool = True,
                 barrier_min: float = -10, barrier_max: float = 10):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Standard minGRU components
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # NAU components (simplified for compilation)
        self.W_add = nn.Parameter(torch.randn(self.dim_inner, self.dim_inner) * 0.01)
        self.W_mul = nn.Parameter(torch.randn(self.dim_inner, self.dim_inner) * 0.01)
        self.gate_mix = nn.Linear(self.dim_inner * 2, self.dim_inner, bias=True)
        
        # Barrier parameters
        self.barrier_strength = nn.Parameter(torch.ones(self.dim_inner) * 0.1)
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
        
        # Flags as tensors for compile compatibility
        self.register_buffer('use_nau_tensor', torch.tensor(float(use_nau)))
        self.register_buffer('use_barriers_tensor', torch.tensor(float(use_barriers)))
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if prev_hidden is None:
            log_hidden = torch.zeros(batch_size, self.dim_inner, device=x.device)
        else:
            log_hidden = torch.log(prev_hidden.abs() + 1e-8)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            hidden_t, gate_t = self.to_hidden_and_gate(x_t).chunk(2, dim=-1)
            
            # Compute NAU update
            h_normal = torch.exp(log_hidden)
            
            # Additive path
            W_add = torch.tanh(self.W_add) * torch.sigmoid(self.W_add)
            add_out = torch.matmul(hidden_t, W_add.T)
            
            # Multiplicative path  
            W_mul = torch.tanh(self.W_mul)
            mul_out = torch.exp(torch.matmul(torch.log(torch.abs(hidden_t) + 1e-8), W_mul.T))
            
            # Gate between paths
            gate_input = torch.cat([hidden_t, h_normal], dim=-1)
            nau_gate = torch.sigmoid(self.gate_mix(gate_input))
            nau_update = nau_gate * add_out + (1 - nau_gate) * mul_out
            
            # Standard minGRU update
            log_standard = self.log_g(hidden_t)
            
            # Mix NAU and standard based on flag
            log_new = self.use_nau_tensor * torch.log(nau_update.abs() + 1e-8) + (1 - self.use_nau_tensor) * log_standard
            
            # Apply gate
            gate_sigmoid = torch.sigmoid(gate_t)
            log_hidden = (1 - gate_sigmoid) * log_hidden + gate_sigmoid * log_new
            
            # Apply barriers
            lower_barrier = -self.barrier_strength / (log_hidden - self.barrier_min + 1e-3)
            upper_barrier = self.barrier_strength / (self.barrier_max - log_hidden + 1e-3)
            barrier_adjustment = self.use_barriers_tensor * (lower_barrier + upper_barrier)
            log_hidden = log_hidden + barrier_adjustment
            log_hidden = torch.clamp(log_hidden, self.barrier_min + 0.1, self.barrier_max - 0.1)
            
            # Output projection
            out = self.to_out(torch.exp(log_hidden))
            outputs.append(out)
        
        output = torch.stack(outputs, dim=1)
        next_hidden = torch.exp(log_hidden)
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
    
    @staticmethod
    def log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
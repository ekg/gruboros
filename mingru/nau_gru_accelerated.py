"""NAU_GRU using torch accelerated operations for sequential processing"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NAUCell(nn.Module):
    """Single NAU cell for use with sequential operations"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined input projection
        self.input_proj = nn.Linear(input_size, hidden_size * 2, bias=False)
        
        # NAU operations
        self.W_add = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.W_mul = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.gate_net = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Project input
        h_new, g = self.input_proj(input).chunk(2, dim=-1)
        
        # NAU operations
        add_path = torch.tanh(self.W_add) @ h_new.T
        mul_path = torch.exp(torch.tanh(self.W_mul) @ torch.log(h_new.abs() + 1e-8).T)
        
        # Gate
        gate_input = torch.cat([h_new, hidden], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        
        # Mix
        h_next = gate * add_path.T + (1 - gate) * mul_path.T
        
        # Apply gating with previous hidden
        g_sigmoid = torch.sigmoid(g)
        return (1 - g_sigmoid) * hidden + g_sigmoid * h_next


class NAU_GRU(nn.Module):
    """NAU-GRU using torch RNN-style operations"""
    def __init__(self, dim: int, expansion_factor: float = 1.5,
                 use_nau: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_nau = use_nau
        
        if use_nau:
            self.cell = NAUCell(dim, self.dim_inner)
        else:
            # Standard GRU cell
            self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
            
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        if not use_nau:
            nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        
        if prev_hidden is None:
            prev_hidden = x.new_zeros(batch_size, self.dim_inner)
        
        if self.use_nau and hasattr(self, 'cell'):
            # Use custom RNN cell with torch acceleration
            outputs = []
            h = prev_hidden
            
            # Unroll using list comprehension (torch can optimize this)
            for t in range(seq_len):
                h = self.cell(x[:, t], h)
                outputs.append(h)
            
            h_seq = torch.stack(outputs, dim=1)
        else:
            # Standard parallel processing
            hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
            h_new = torch.where(
                hidden >= 0,
                (F.relu(hidden) + 0.5).log(),
                -F.softplus(-hidden)
            )
            gate_sigmoid = torch.sigmoid(gate)
            
            # Simple mixing
            if seq_len == 1:
                h_seq = (1 - gate_sigmoid) * prev_hidden.unsqueeze(1) + gate_sigmoid * torch.exp(h_new)
            else:
                h_seq = gate_sigmoid * torch.exp(h_new)
        
        # Output projection
        output = self.to_out(h_seq)
        next_hidden = h_seq[:, -1]
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
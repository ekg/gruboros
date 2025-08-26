"""NAU-GRU using torch RNN cell pattern for efficient sequential processing"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit, Tensor
from typing import Tuple


@jit.script
def gru_cell(h: Tensor, h_t: Tensor, g_t: Tensor) -> Tensor:
    """JIT-compiled GRU cell operation"""
    # Activation
    h_new = torch.where(
        h_t >= 0,
        (F.relu(h_t) + 0.5).log(),
        -F.softplus(-h_t)
    )
    
    # Gate mixing
    g_sigmoid = torch.sigmoid(g_t)
    h_log = torch.log(torch.abs(h) + 1e-8)
    h_log = (1 - g_sigmoid) * h_log + g_sigmoid * h_new
    return torch.exp(h_log)


class NAU_GRU(nn.Module):
    """Efficient sequential GRU using custom cell"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Combined projection for efficiency
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        # Get dimensions
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Single projection for all timesteps
        combined = self.to_hidden_and_gate(x)
        hidden_seq, gate_seq = combined.chunk(2, dim=-1)
        
        # Initialize state
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            h = prev_hidden
        
        # Process sequence
        output_list = []
        
        # Use the JIT-compiled cell
        for t in range(seq_len):
            h = gru_cell(h, hidden_seq[:, t], gate_seq[:, t])
            output_list.append(h)
        
        # Stack outputs efficiently
        h_outputs = torch.stack(output_list, dim=1)
        
        # Project to output dimension
        output = self.to_out(h_outputs)
        
        # Return
        if not return_next_prev_hidden:
            return output
        return output, h
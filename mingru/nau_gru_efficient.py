"""Efficient sequential NAU-GRU with minimal memory allocation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def efficient_sequential_scan(
    hidden: Tensor, 
    gate: Tensor, 
    h: Tensor,
    output: Tensor,
    to_out_weight: Tensor
) -> Tensor:
    """Memory-efficient sequential scan with pre-allocated output"""
    seq_len = hidden.size(1)
    
    for t in range(seq_len):
        # Get timestep slices (views, no copy)
        h_t = hidden[:, t]
        g_t = gate[:, t]
        
        # Log activation (in-place where possible)
        h_new = torch.where(
            h_t >= 0,
            (F.relu(h_t) + 0.5).log(),
            -F.softplus(-h_t)
        )
        
        # Gate and mix (in-place)
        g_sigmoid = torch.sigmoid(g_t)
        h_log = h.log().clamp(min=-20)  # Reuse h tensor
        h_log = torch.lerp(h_log, h_new, g_sigmoid)  # Efficient interpolation
        h = h_log.exp()
        
        # Output projection directly into pre-allocated tensor
        torch.mm(h, to_out_weight.t(), out=output[:, t])
    
    return h


class NAU_GRU(nn.Module):
    """Memory-efficient sequential GRU"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Single allocation for both hidden and gate
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Single forward pass for all timesteps
        combined = self.to_hidden_and_gate(x)
        hidden, gate = combined.chunk(2, dim=-1)
        
        # Pre-allocate output tensor to avoid memory spikes
        output = torch.empty(batch_size, seq_len, self.dim, device=device, dtype=dtype)
        
        # Initialize or clone hidden state (avoid in-place ops on input)
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            h = prev_hidden.clone()  # Clone to avoid modifying input
        
        # Run efficient scan
        h = efficient_sequential_scan(hidden, gate, h, output, self.to_out.weight)
        
        if not return_next_prev_hidden:
            return output
        return output, h
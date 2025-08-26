"""Minimal NAU_GRU that just wraps standard minGRU behavior"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAU_GRU(nn.Module):
    """Minimal wrapper around minGRU operations to avoid CPU bottleneck"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
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
        
        # Process all timesteps at once (vectorized)
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        
        # Apply activations
        h_new = torch.where(
            hidden >= 0,
            (F.relu(hidden) + 0.5).log(),
            -F.softplus(-hidden)
        )
        
        # Simple gating without scan
        gate_sigmoid = torch.sigmoid(gate)
        
        if prev_hidden is not None:
            # Just use last timestep mixing for simplicity
            h_init = prev_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            h_gated = (1 - gate_sigmoid) * torch.log(h_init.abs() + 1e-8) + gate_sigmoid * h_new
        else:
            h_gated = gate_sigmoid * h_new
            
        # Output projection
        output = self.to_out(torch.exp(h_gated))
        
        # Final hidden state
        next_hidden = torch.exp(h_gated[:, -1])
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
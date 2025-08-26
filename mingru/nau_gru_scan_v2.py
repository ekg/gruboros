"""NAU-GRU using torch scan operations for efficient sequential processing"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NAU_GRU(nn.Module):
    """Sequential GRU using torch.jit and functional operations"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Standard components
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Process all timesteps at once
        combined = self.to_hidden_and_gate(x)
        hidden, gate = combined.chunk(2, dim=-1)
        
        # Initialize hidden state
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device, dtype=dtype)
        else:
            h = prev_hidden
        
        # Use torch.nn.utils.rnn.pad_sequence or custom scan
        outputs = []
        
        # Unroll manually but efficiently
        for t in range(seq_len):
            # Slicing is efficient and compile-friendly
            h_t = hidden[:, t]
            g_t = gate[:, t]
            
            # Activation
            h_new = torch.where(
                h_t >= 0,
                (F.relu(h_t) + 0.5).log(),
                -F.softplus(-h_t)
            )
            
            # Mix with gate
            g_sigmoid = torch.sigmoid(g_t)
            h_log = torch.log(torch.abs(h) + 1e-8)
            h_log = (1 - g_sigmoid) * h_log + g_sigmoid * h_new
            h = torch.exp(h_log)
            
            outputs.append(h)
        
        # Stack and project
        h_seq = torch.stack(outputs, dim=1)
        output = self.to_out(h_seq)
        
        # Final hidden
        next_hidden = h
        
        if not return_next_prev_hidden:
            return output
        return output, next_hidden
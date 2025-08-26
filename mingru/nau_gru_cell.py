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
        return output, h
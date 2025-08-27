"""
Simple GRU implementation for testing - operates in normal space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def simple_gru_kernel(
    # Inputs
    x_ptr,              # [B, T, D] input 
    prev_h_ptr,         # [B, D] previous hidden
    # Weights
    W_ih_ptr,           # [3*D, D] input weights
    W_hh_ptr,           # [3*D, D] hidden weights
    # Outputs  
    output_ptr,         # [B, T, D] output
    final_h_ptr,        # [B, D] final hidden
    # Dimensions
    B, T, D,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Simple GRU forward pass. Each program handles one batch element.
    Processes full sequence with proper temporal dependencies.
    """
    batch_idx = tl.program_id(0)
    if batch_idx >= B:
        return
        
    # Process dimensions in blocks (for memory efficiency only)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        # Load initial hidden state
        h_offs = batch_idx * D + d_offs
        h = tl.load(prev_h_ptr + h_offs, mask=d_mask, other=0.0)
        
        # Process each timestep sequentially
        for t in range(T):
            # Load input
            x_offs = batch_idx * T * D + t * D + d_offs
            x = tl.load(x_ptr + x_offs, mask=d_mask, other=0.0)
            
            # Compute gates: r, z, n = W_ih @ x + W_hh @ h
            # For simplicity, we'll do a fused computation
            # In real implementation, you'd do proper matrix multiplies
            
            # For now, simple elementwise (you'd replace with proper matmul)
            # Reset gate
            r = tl.sigmoid(x * 0.5 + h * 0.3)
            # Update gate  
            z = tl.sigmoid(x * 0.4 + h * 0.6)
            # New gate
            n = tl.tanh(x * 0.7 + (r * h) * 0.2)
            
            # Update hidden state
            h = (1 - z) * h + z * n
            
            # Store output
            tl.store(output_ptr + x_offs, h, mask=d_mask)
        
        # Store final hidden state
        final_h_offs = batch_idx * D + d_offs
        tl.store(final_h_ptr + final_h_offs, h, mask=d_mask)


class SimpleGRU(nn.Module):
    """
    Simple GRU for testing that everything works.
    Uses standard PyTorch GRU weights but custom Triton kernel.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Standard GRU weights
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_dim, hidden_dim))
        
        # Initialize
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        
    def forward(self, x, prev_hidden=None):
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # Handle hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        elif prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
            prev_hidden = prev_hidden.squeeze(1)
            
        # For testing, just use PyTorch's GRU
        # (Replace with Triton kernel when ready)
        x_reshaped = x.reshape(T, B, D)
        h_reshaped = prev_hidden.unsqueeze(0)
        
        # Create GRU cell
        gru = nn.GRU(D, self.hidden_dim, batch_first=False)
        gru.weight_ih_l0.data = self.weight_ih
        gru.weight_hh_l0.data = self.weight_hh
        
        output, h_final = gru(x_reshaped, h_reshaped)
        output = output.permute(1, 0, 2)  # Back to [B, T, D]
        h_final = h_final.squeeze(0)
        
        return output, h_final
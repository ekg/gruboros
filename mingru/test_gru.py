"""
Test GRU implementation that matches minGRU interface
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def test_gru_kernel(
    x_ptr, h_ptr, g_ptr,      # Inputs: input, hidden, gate
    prev_h_ptr,               # Previous hidden state  
    output_ptr,               # Output 
    final_h_ptr,              # Final hidden state
    B, T, D,                  # Dimensions
    BLOCK_D: tl.constexpr,    # Block size for D dimension
):
    """
    Simple GRU-like kernel that matches NAU_GRU interface.
    Processes one batch element at a time.
    """
    batch_idx = tl.program_id(0)
    if batch_idx >= B:
        return
    
    # Process all dimensions together (in blocks for memory efficiency)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        
        # Load initial hidden state
        h_prev_offset = batch_idx * D + d_offs
        h_state = tl.load(prev_h_ptr + h_prev_offset, mask=mask, other=0.0)
        
        # Forward through time
        for t in range(T):
            # Load inputs for this timestep
            offset = batch_idx * T * D + t * D + d_offs
            x_t = tl.load(x_ptr + offset, mask=mask, other=0.0)
            h_t = tl.load(h_ptr + offset, mask=mask, other=0.0)
            g_t = tl.load(g_ptr + offset, mask=mask, other=0.0)
            
            # Simple tanh activation on new input
            # Triton doesn't have tanh, so use approximation: tanh(x) ≈ x for small x
            # Or use the formula: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            exp_2x = tl.exp(2.0 * h_t)
            h_new = (exp_2x - 1.0) / (exp_2x + 1.0)
            
            # Gate (sigmoid)
            gate = tl.sigmoid(g_t)
            
            # GRU update: h = (1-g) * h_prev + g * h_new
            h_state = (1.0 - gate) * h_state + gate * h_new
            
            # Store output
            tl.store(output_ptr + offset, h_state, mask=mask)
        
        # Store final hidden state
        final_offset = batch_idx * D + d_offs
        tl.store(final_h_ptr + final_offset, h_state, mask=mask)


class TestGRU(nn.Module):
    """Test GRU that matches the NAU_GRU interface"""
    
    def __init__(self, dim, expansion_factor=1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Same interface as NAU_GRU
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        # Simple initialization
        nn.init.normal_(self.to_hidden_and_gate.weight, std=0.02)
        nn.init.zeros_(self.to_out.weight)  # Start as identity
        
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project input to hidden and gate
        combined = self.to_hidden_and_gate(x)
        h, g = combined.chunk(2, dim=-1)
        h = h.contiguous()
        g = g.contiguous()
        
        # Handle previous hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            # Handle [B, 1, D] format from minGRU
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            # Handle batch size mismatch
            if prev_hidden.shape[0] != B:
                prev_hidden = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            else:
                prev_hidden = prev_hidden.contiguous()
        
        # Allocate output tensors
        output = torch.empty(B, T, self.dim_inner, device=device, dtype=dtype)
        final_hidden = torch.empty(B, self.dim_inner, device=device, dtype=dtype)
        
        # Launch kernel
        BLOCK_D = min(256, triton.next_power_of_2(self.dim_inner))
        grid = (B,)
        
        test_gru_kernel[grid](
            x, h, g,
            prev_hidden,
            output,
            final_hidden,
            B, T, self.dim_inner,
            BLOCK_D=BLOCK_D
        )
        
        # Project output back
        output = self.to_out(output)
        
        if return_next_prev_hidden:
            # Return in [B, 1, D] format to match minGRU
            return output, final_hidden.unsqueeze(1).contiguous()
        return output
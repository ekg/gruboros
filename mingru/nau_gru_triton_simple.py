import torch
import triton
import triton.language as tl

@triton.jit
def simple_gru_kernel(
    x_ptr, h_prev_ptr,
    output_ptr,
    seq_len, dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Process dimensions in blocks
    dim_idx = tl.arange(0, BLOCK_SIZE)
    mask = dim_idx < dim
    
    # Load initial hidden state
    h_offset = batch_idx * dim + dim_idx
    h = tl.load(h_prev_ptr + h_offset, mask=mask, other=0.0)
    
    # Process sequence
    for t in range(seq_len):
        # Load input
        x_offset = batch_idx * seq_len * dim + t * dim + dim_idx
        x_t = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Simple GRU update (no complex log operations)
        # Just use tanh activation
        h_new = tl.tanh(x_t)
        
        # Simple gate (sigmoid of x)
        gate = tl.sigmoid(x_t)
        
        # Update hidden state
        h = (1.0 - gate) * h + gate * h_new
        
        # Store output
        tl.store(output_ptr + x_offset, h, mask=mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Simple linear layers
        self.to_hidden_and_gate = torch.nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize with smaller values for stability
        with torch.no_grad():
            self.to_hidden_and_gate.weight.uniform_(-0.01, 0.01)
            self.to_out.weight.uniform_(-0.01, 0.01)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project input
        combined = self.to_hidden_and_gate(x)
        h_and_g = combined.view(B, T, 2, self.dim_inner)
        
        # Initialize hidden
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        
        # Ensure contiguous
        h_and_g = h_and_g.contiguous()
        prev_hidden = prev_hidden.contiguous()
        
        # Output tensor
        h_output = torch.empty(B, T, self.dim_inner, device=device, dtype=dtype)
        
        # Launch simple kernel
        grid = (B,)
        BLOCK_SIZE = min(1024, triton.next_power_of_2(self.dim_inner))
        
        # Combine h and g into single tensor for simpler kernel
        combined_for_kernel = h_and_g[:, :, 0, :] + h_and_g[:, :, 1, :]
        
        simple_gru_kernel[grid](
            combined_for_kernel, prev_hidden,
            h_output,
            T, self.dim_inner,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project output
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            return output, h_output[:, -1, :].contiguous()
        return output
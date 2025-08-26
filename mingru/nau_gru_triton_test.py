import torch
import triton
import triton.language as tl

@triton.jit
def nau_gru_kernel_v2(
    h_ptr, g_ptr, h_prev_ptr,
    output_ptr,
    seq_len: tl.constexpr, 
    dim_inner: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process one batch element
    batch_idx = tl.program_id(0)
    
    # Process all dimensions for this batch in blocks
    for block_start in tl.static_range(0, dim_inner, BLOCK_SIZE):
        # Get dimension indices for this block
        dim_idx = block_start + tl.arange(0, BLOCK_SIZE)
        mask = dim_idx < dim_inner
        
        # Load initial hidden state
        h_prev_offset = batch_idx * dim_inner + dim_idx  
        h_prev = tl.load(h_prev_ptr + h_prev_offset, mask=mask, other=0.0)
        
        # Process each timestep
        for t in tl.static_range(seq_len):
            # Calculate offset
            offset = batch_idx * seq_len * dim_inner + t * dim_inner + dim_idx
            
            # Load inputs
            h_t = tl.load(h_ptr + offset, mask=mask)
            g_t = tl.load(g_ptr + offset, mask=mask)
            
            # Simple update without complex log operations
            # Just linear interpolation with sigmoid gate
            g_sigmoid = tl.sigmoid(g_t)
            h_prev = (1 - g_sigmoid) * h_prev + g_sigmoid * h_t
            
            # Store
            tl.store(output_ptr + offset, h_prev, mask=mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Matching JIT version exactly
        self.to_hidden_and_gate = torch.nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        torch.nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        torch.nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project input  
        combined = self.to_hidden_and_gate(x)
        h, g = combined.chunk(2, dim=-1)
        
        # Ensure contiguous
        h = h.contiguous()
        g = g.contiguous()
        
        # Initialize hidden
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        prev_hidden = prev_hidden.contiguous()
        
        # Output tensor
        h_output = torch.empty(B, T, self.dim_inner, device=device, dtype=dtype)
        
        # Fixed dimensions for kernel
        BLOCK_SIZE = 128
        grid = (B,)
        
        # Launch kernel with fixed seq_len
        nau_gru_kernel_v2[grid](
            h, g, prev_hidden,
            h_output,
            T, self.dim_inner,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project output
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            return output, h_output[:, -1, :].contiguous()
        return output
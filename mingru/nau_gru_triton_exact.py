import torch
import triton
import triton.language as tl

@triton.jit
def nau_gru_exact_kernel(
    h_ptr, g_ptr, h_prev_ptr,
    output_ptr,
    batch_size, seq_len, dim_inner,
    BLOCK_SIZE: tl.constexpr,
):
    # Get position in grid
    pid = tl.program_id(0)
    
    # Handle one batch element's dimensions
    batch_idx = pid
    if batch_idx >= batch_size:
        return
        
    # Process dimensions in chunks
    for dim_start in range(0, dim_inner, BLOCK_SIZE):
        dim_idx = dim_start + tl.arange(0, BLOCK_SIZE)
        mask = dim_idx < dim_inner
        
        # Load initial hidden state
        h_prev_offset = batch_idx * dim_inner + dim_idx
        h_prev = tl.load(h_prev_ptr + h_prev_offset, mask=mask, other=1e-8).to(tl.float32)
        
        # Process sequence
        for t in range(seq_len):
            offset = batch_idx * seq_len * dim_inner + t * dim_inner + dim_idx
            
            # Load inputs as float32
            h_t = tl.load(h_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            g_t = tl.load(g_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            
            # Match JIT version exactly:
            # h_new = torch.where(h_t >= 0, (F.relu(h_t) + 0.5).log(), -F.softplus(-h_t))
            
            # For positive h_t: log(relu(h_t) + 0.5) = log(h_t + 0.5)
            h_t_pos = tl.maximum(h_t, 0.0)
            h_new_pos = tl.log(h_t_pos + 0.5)
            
            # For negative h_t: -softplus(-h_t) = -log(1 + exp(-h_t))
            h_t_neg_abs = tl.abs(tl.minimum(h_t, 0.0))
            h_new_neg = -tl.log(1.0 + tl.exp(-h_t_neg_abs))
            
            # Combine
            h_new = tl.where(h_t >= 0, h_new_pos, h_new_neg)
            
            # Gate
            g_sigmoid = tl.sigmoid(g_t)
            
            # Log space update - match JIT exactly
            h_prev_abs = tl.abs(h_prev) + 1e-8
            h_log = tl.log(h_prev_abs)
            h_log_new = (1.0 - g_sigmoid) * h_log + g_sigmoid * h_new
            
            # Clamp
            h_log_new = tl.minimum(tl.maximum(h_log_new, -20.0), 20.0)
            h_prev = tl.exp(h_log_new)
            
            # Store as original dtype
            tl.store(output_ptr + offset, h_prev.to(h_ptr.dtype.element_ty), mask=mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Match JIT version exactly
        self.to_hidden_and_gate = torch.nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim_inner, dim, bias=False)
        
        torch.nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        torch.nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype
        
        # Combined projection
        combined = self.to_hidden_and_gate(x)
        h, g = combined.chunk(2, dim=-1)
        
        # Make contiguous
        h = h.contiguous()
        g = g.contiguous()
        
        # Initialize hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            prev_hidden = prev_hidden.contiguous()
            
        # Output tensor
        h_output = torch.empty(B, T, self.dim_inner, device=device, dtype=dtype)
        
        # Kernel config
        BLOCK_SIZE = 256
        grid = (B,)
        
        # Launch kernel
        nau_gru_exact_kernel[grid](
            h, g, prev_hidden,
            h_output,
            B, T, self.dim_inner,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project back
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            return output, h_output[:, -1, :].contiguous()
        return output
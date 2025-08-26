import torch
import triton
import triton.language as tl

@triton.jit
def nau_gru_forward_kernel(
    h_ptr, g_ptr, h_prev_ptr,
    output_ptr,
    seq_len, dim_inner,
    stride_batch, stride_seq, stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Process dimensions in chunks
    for dim_start in range(0, dim_inner, BLOCK_SIZE):
        # Create dimension indices
        dim_indices = dim_start + tl.arange(0, BLOCK_SIZE)
        dim_mask = dim_indices < dim_inner
        
        # Load initial hidden state
        h_prev_offset = batch_idx * stride_batch + dim_indices
        h_prev = tl.load(h_prev_ptr + h_prev_offset, mask=dim_mask, other=0.0)
        
        # Process sequence
        for t in range(seq_len):
            # Calculate offsets with explicit strides
            seq_offset = (batch_idx * stride_batch + 
                         t * stride_seq + 
                         dim_indices)
            
            # Load h_t and g_t
            h_t = tl.load(h_ptr + seq_offset, mask=dim_mask, other=0.0)
            g_t = tl.load(g_ptr + seq_offset, mask=dim_mask, other=0.0)
            
            # NAU computation
            # Safe log transformation
            h_pos = tl.maximum(h_t, 0.0)
            h_neg = tl.minimum(h_t, 0.0)
            
            h_new_pos = tl.log(h_pos + 0.5)
            h_new_neg = -tl.log(1.0 + tl.exp(-tl.abs(h_neg)))
            h_new = tl.where(h_t >= 0, h_new_pos, h_new_neg)
            
            # Gate
            g_sigmoid = tl.sigmoid(g_t)
            
            # Update in log space
            h_prev_log = tl.log(tl.abs(h_prev) + 1e-8)
            h_log_new = (1 - g_sigmoid) * h_prev_log + g_sigmoid * h_new
            
            # Clamp and exp
            h_log_new = tl.minimum(tl.maximum(h_log_new, -20.0), 20.0)
            h_prev = tl.exp(h_log_new)
            
            # Store
            tl.store(output_ptr + seq_offset, h_prev, mask=dim_mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, use_nau=True, use_barriers=False,
                 barrier_min=-5.0, barrier_max=5.0, barrier_strength=0.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_in = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_nau = use_nau
        
        # Linear transformations  
        self.to_hidden_and_gate = torch.nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim_inner, dim, bias=False)
        
        # Initialize
        torch.nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        torch.nn.init.xavier_uniform_(self.to_out.weight)
        
        # NAU-specific parameters
        self.barrier_strength = barrier_strength
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, C = x.shape
        
        # Combined projection for efficiency
        combined = self.to_hidden_and_gate(x)
        h, g = combined.chunk(2, dim=-1)
        
        # Initialize hidden state - handle batch size changes
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=x.device, dtype=x.dtype)
        elif prev_hidden.shape[0] != B:
            # Batch size changed - reinitialize
            prev_hidden = torch.zeros(B, self.dim_inner, device=x.device, dtype=x.dtype)
        
        # Make sure tensors are contiguous
        h = h.contiguous()
        g = g.contiguous()
        prev_hidden = prev_hidden.contiguous()
        
        # Create output tensor for hidden states (dim_inner)
        h_output = torch.empty_like(h)
        
        # Calculate strides
        stride_batch = T * self.dim_inner
        stride_seq = self.dim_inner
        stride_dim = 1
        
        # Choose block size
        BLOCK_SIZE = min(128, triton.next_power_of_2(self.dim_inner))
        
        # Launch kernel - one program per batch
        grid = (B,)
        
        nau_gru_forward_kernel[grid](
            h, g, prev_hidden,
            h_output,
            T, self.dim_inner,
            stride_batch, stride_seq, stride_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project back to output dimension
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            # Extract final hidden state (from h_output, not projected output)
            final_hidden = h_output[:, -1, :].contiguous()
            return output, final_hidden
        
        return output
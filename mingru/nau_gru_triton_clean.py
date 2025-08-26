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
        
        # Parameters
        self.W_h = torch.nn.Parameter(torch.randn(self.dim_in, self.dim_inner) * 0.02)
        self.W_g = torch.nn.Parameter(torch.randn(self.dim_in, self.dim_inner) * 0.02)
        
        # NAU-specific parameters
        self.nau_gate = torch.nn.Parameter(torch.zeros(self.dim_inner))
        self.barrier_strength = barrier_strength
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
    
    def forward(self, x, prev_hidden=None, doc_boundaries=None):
        B, T, C = x.shape
        
        # Linear transformations
        h = torch.einsum('btc,cd->btd', x, self.W_h)
        g = torch.einsum('btc,cd->btd', x, self.W_g)
        
        # Initialize hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.dim_inner, device=x.device, dtype=x.dtype)
        
        # Make sure tensors are contiguous
        h = h.contiguous()
        g = g.contiguous()
        prev_hidden = prev_hidden.contiguous()
        
        # Create output tensor
        output = torch.empty_like(h)
        
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
            output,
            T, self.dim_inner,
            stride_batch, stride_seq, stride_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Extract final hidden state
        final_hidden = output[:, -1, :].contiguous()
        
        return output, final_hidden
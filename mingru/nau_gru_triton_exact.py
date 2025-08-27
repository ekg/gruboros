import torch
import triton
import triton.language as tl

@triton.jit
def nau_gru_exact_kernel(
    h_ptr, g_ptr, h_prev_ptr,
    output_ptr,
    h_log_final_ptr,
    batch_size, seq_len, dim_inner,
    dim_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid-strided pattern: each kernel handles specific dimensions across all batches
    pid = tl.program_id(0)
    
    # Calculate batch and dimension block
    num_dim_blocks = tl.cdiv(dim_inner, BLOCK_SIZE)
    batch_idx = pid // num_dim_blocks
    dim_block_idx = pid % num_dim_blocks
    
    if batch_idx >= batch_size or dim_block_idx >= num_dim_blocks:
        return
    
    # This kernel handles these specific dimensions
    dim_start = dim_block_idx * BLOCK_SIZE
    dim_idx = dim_start + tl.arange(0, BLOCK_SIZE)
    mask = dim_idx < dim_inner
    
    # Load initial hidden state for THESE dimensions
    h_prev_offset = batch_idx * dim_inner + dim_idx
    h_log = tl.load(h_prev_ptr + h_prev_offset, mask=mask, other=-20.0).to(tl.float32)
    
    # Safety check for empty sequences
    if seq_len <= 0:
        return
    
    # Process sequence for THESE dimensions
    for t in range(seq_len):
        offset = batch_idx * seq_len * dim_inner + t * dim_inner + dim_idx
        
        # Load inputs
        h_t = tl.load(h_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        
        # Match minGRU exactly: log_g activation
        h_t_pos = tl.maximum(h_t, 0.0)
        h_new_pos = tl.log(h_t_pos + 0.5)
        h_new_neg = -tl.log(1.0 + tl.exp(-h_t))
        log_h_new = tl.where(h_t >= 0, h_new_pos, h_new_neg)
        
        # Gate - match minGRU exactly
        gate_sigmoid = tl.sigmoid(g_t)
        eps = 1e-8
        
        # Match minGRU's gate computation
        log_gate = tl.log(tl.maximum(gate_sigmoid, eps))
        log_one_minus_gate = tl.log(tl.maximum(1.0 - gate_sigmoid, eps))
        
        # Combine with log-sum-exp
        term1 = log_one_minus_gate + h_log
        term2 = log_gate + log_h_new
        max_val = tl.maximum(term1, term2)
        h_log = max_val + tl.log(tl.exp(term1 - max_val) + tl.exp(term2 - max_val))
        
        # Output - clamp before exp to prevent overflow
        h_log_clamped = tl.minimum(tl.maximum(h_log, -20.0), 20.0)
        h_output = tl.exp(h_log_clamped)
        tl.store(output_ptr + offset, h_output.to(h_ptr.dtype.element_ty), mask=mask)
    
    # Store final hidden state for THESE dimensions
    if seq_len > 0:  # Only store if we processed timesteps
        final_offset = batch_idx * dim_inner + dim_idx
        tl.store(h_log_final_ptr + final_offset, h_log.to(h_ptr.dtype.element_ty), mask=mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Match JIT version exactly
        self.to_hidden_and_gate = torch.nn.Linear(dim, self.dim_inner * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim_inner, dim, bias=False)
        
        # Match minLM's careful initialization
        import math
        std = 0.02 / math.sqrt(dim)
        torch.nn.init.normal_(self.to_hidden_and_gate.weight, mean=0.0, std=std)
        # CRITICAL: Zero-initialize output for residual identity
        torch.nn.init.constant_(self.to_out.weight, 0.0)
    
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
        
        # Initialize hidden state IN LOG SPACE
        if prev_hidden is None:
            # Start at -2.0 instead of -20.0: exp(-2) ≈ 0.135 vs exp(-20) ≈ 2e-9
            prev_hidden = torch.full((B, self.dim_inner), -2.0, device=device, dtype=dtype)
        else:
            # Handle minGRU's [B, 1, D] format!
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            
            # Ensure prev_hidden has correct batch size
            if prev_hidden.shape[0] != B:
                # Batch size changed (e.g., validation) - reinitialize
                prev_hidden = torch.full((B, self.dim_inner), -20.0, device=device, dtype=dtype)
            else:
                # prev_hidden is already in log space, don't convert
                prev_hidden = prev_hidden.contiguous()
            
        # Output tensors
        h_output = torch.empty(B, T, self.dim_inner, device=device, dtype=dtype)
        h_log_final = torch.empty(B, self.dim_inner, device=device, dtype=dtype)
        
        # Kernel config  
        BLOCK_SIZE = min(256, triton.next_power_of_2(self.dim_inner))
        num_dim_blocks = triton.cdiv(self.dim_inner, BLOCK_SIZE)
        grid = (B * num_dim_blocks,)  # Grid-strided: each kernel handles specific dims
        
        # Launch kernel
        nau_gru_exact_kernel[grid](
            h, g, prev_hidden,
            h_output,
            h_log_final,
            B, T, self.dim_inner,
            dim_offset=0,  # Not used but required for signature
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project back
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            # Return the actual log-space hidden state from kernel
            # CRITICAL: Must be [B, 1, D] to match minGRU's format!
            return output, h_log_final.unsqueeze(1).contiguous()
        return output
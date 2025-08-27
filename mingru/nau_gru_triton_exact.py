import torch
import triton
import triton.language as tl

@triton.jit
def nau_gru_exact_kernel(
    h_ptr, g_ptr, h_prev_ptr,
    output_ptr,
    h_log_final_ptr,  # ADD: output buffer for final log-space hidden state
    batch_size, seq_len, dim_inner,
    use_barriers: tl.constexpr,
    barrier_min: tl.constexpr, 
    barrier_max: tl.constexpr,
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
        
        # Load initial hidden state - IT'S ALREADY IN LOG SPACE
        h_prev_offset = batch_idx * dim_inner + dim_idx
        h_log = tl.load(h_prev_ptr + h_prev_offset, mask=mask, other=-20.0).to(tl.float32)
        
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
            
            # Gate log probabilities
            log_g = -tl.log(1.0 + tl.exp(-g_t))        # log(sigmoid(g_t))
            log_one_minus_g = -tl.log(1.0 + tl.exp(g_t))  # log(1 - sigmoid(g_t))
            
            # Log-sum-exp (h_log is ALREADY in log space, no log() needed!)
            term1 = log_one_minus_g + h_log  # log((1-g)*exp(h_log))
            term2 = log_g + h_new            # log(g*exp(h_new))
            
            # Stable log-sum-exp
            max_val = tl.maximum(term1, term2)
            h_log = max_val + tl.log(tl.exp(term1 - max_val) + tl.exp(term2 - max_val))
            
            # Apply energy barriers to prevent explosion
            if use_barriers:
                # Soft barrier forces that grow exponentially near boundaries
                # Lower barrier: repulsive force when h_log approaches barrier_min
                lower_force = 0.5 * tl.exp(barrier_min - h_log + 5.0)  # +5 to activate earlier
                # Upper barrier: repulsive force when h_log approaches barrier_max  
                upper_force = 0.5 * tl.exp(h_log - barrier_max + 5.0)
                
                # Apply corrections - push away from barriers
                h_log = h_log + lower_force - upper_force
                
                # Hard clamp as final safety
                h_log = tl.minimum(tl.maximum(h_log, barrier_min), barrier_max)
            else:
                # Simple clamp without barriers
                h_log = tl.minimum(tl.maximum(h_log, -20.0), 20.0)
            
            # Only exp() for output, keep h_log for next iteration
            h_output = tl.exp(h_log)
            
            # Store exp(h_log) as original dtype
            tl.store(output_ptr + offset, h_output.to(h_ptr.dtype.element_ty), mask=mask)
        
        # CRITICAL: Store final h_log (not exp!) for next chunk
        final_offset = batch_idx * dim_inner + dim_idx
        tl.store(h_log_final_ptr + final_offset, h_log.to(h_ptr.dtype.element_ty), mask=mask)

class NAU_GRU(torch.nn.Module):
    def __init__(self, dim, expansion_factor=1.5, use_barriers=True, barrier_min=-10, barrier_max=10, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.use_barriers = use_barriers
        self.barrier_min = barrier_min
        self.barrier_max = barrier_max
        
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
        grid = (B,)
        
        # Launch kernel
        nau_gru_exact_kernel[grid](
            h, g, prev_hidden,
            h_output,
            h_log_final,  # Pass buffer for final log-space hidden
            B, T, self.dim_inner,
            use_barriers=self.use_barriers,
            barrier_min=self.barrier_min,
            barrier_max=self.barrier_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Project back
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            # Return the ACTUAL log-space hidden state (no log(exp()) conversion!)
            return output, h_log_final.contiguous()
        return output
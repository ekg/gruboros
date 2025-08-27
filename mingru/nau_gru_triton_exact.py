import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def gru_kernel(
    # Inputs
    h_ptr, g_ptr,           # [B, T, D] input and gate
    h_prev_ptr,             # [B, D] previous hidden state (log space)
    # Outputs
    output_ptr,             # [B, T, D] output
    h_final_ptr,            # [B, D] final hidden state (log space)
    # Dimensions
    batch_size, seq_len, hidden_dim,
    # Block size
    BLOCK_DIM: tl.constexpr
):
    """
    Regular GRU that processes ALL dimensions together.
    Each program handles one batch element.
    """
    # One program per batch element
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Process ALL hidden dimensions in blocks
    # This is just for memory efficiency - they all update together
    for d_start in range(0, hidden_dim, BLOCK_DIM):
        d_offs = d_start + tl.arange(0, BLOCK_DIM)
        d_mask = d_offs < hidden_dim
        
        # Load previous hidden state (in log space)
        h_prev_offs = batch_idx * hidden_dim + d_offs
        h_log = tl.load(h_prev_ptr + h_prev_offs, mask=d_mask, other=-2.0)
        
        # Process sequence timestep by timestep
        for t in range(seq_len):
            # Offset for this timestep
            offset = batch_idx * seq_len * hidden_dim + t * hidden_dim + d_offs
            
            # Load inputs
            h_t = tl.load(h_ptr + offset, mask=d_mask, other=0.0)
            g_t = tl.load(g_ptr + offset, mask=d_mask, other=0.0)
            
            # MinGRU-style activation: log_g(x)
            h_pos = tl.maximum(h_t, 0.0)
            h_new_pos = tl.log(h_pos + 0.5)
            h_new_neg = -tl.log1p(tl.exp(-tl.abs(tl.minimum(h_t, 0.0))))
            h_new = tl.where(h_t >= 0, h_new_pos, h_new_neg)
            
            # Gate (sigmoid)
            g_sigmoid = tl.sigmoid(g_t)
            
            # Compute in log space for stability
            eps = 1e-8
            log_gate = tl.log(tl.maximum(g_sigmoid, eps))
            log_one_minus_gate = tl.log(tl.maximum(1.0 - g_sigmoid, eps))
            
            # GRU update: h = (1-g) * h_prev + g * h_new
            # In log space: log(exp(a) + exp(b))
            term1 = log_one_minus_gate + h_log
            term2 = log_gate + h_new
            
            # Stable log-sum-exp
            max_term = tl.maximum(term1, term2)
            h_log = max_term + tl.log(
                tl.exp(term1 - max_term) + tl.exp(term2 - max_term)
            )
            
            # Clamp for stability
            h_log = tl.minimum(tl.maximum(h_log, -20.0), 20.0)
            
            # Store output (exp of log-space hidden)
            h_out = tl.exp(h_log)
            tl.store(output_ptr + offset, h_out, mask=d_mask)
        
        # Store final hidden state in log space
        h_final_offs = batch_idx * hidden_dim + d_offs
        tl.store(h_final_ptr + h_final_offs, h_log, mask=d_mask)

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
        
        # Launch GRU kernel
        grid = (B,)  # One program per batch element
        BLOCK_DIM = 256  # Process dims in blocks of 256
        
        gru_kernel[grid](
            h, g, prev_hidden,
            h_output, h_log_final,
            B, T, self.dim_inner,
            BLOCK_DIM=BLOCK_DIM
        )
        
        # Project back
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            # Return the actual log-space hidden state from kernel
            # CRITICAL: Must be [B, 1, D] to match minGRU's format!
            return output, h_log_final.unsqueeze(1).contiguous()
        return output
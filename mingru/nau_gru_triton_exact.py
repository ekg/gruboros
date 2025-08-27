import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# TEMPORARY: Process in PyTorch until we fix the Triton kernel
def nau_gru_sequential_forward(h, g, prev_hidden, B, T, dim_inner):
    """
    Pure PyTorch sequential forward - processes ALL dimensions together
    """
    device = h.device
    dtype = h.dtype
    
    # Initialize or use previous hidden state (in log space)
    if prev_hidden is None:
        h_log = torch.full((B, dim_inner), -2.0, device=device, dtype=dtype)
    else:
        h_log = prev_hidden
        if h_log.dim() == 3:  # Handle [B, 1, D] format
            h_log = h_log.squeeze(1)
    
    outputs = []
    
    # Sequential forward pass
    for t in range(T):
        h_t = h[:, t, :]  # [B, D]
        g_t = g[:, t, :]  # [B, D]
        
        # Activation (matches minGRU)
        log_h_new = torch.where(
            h_t >= 0,
            torch.log(torch.relu(h_t) + 0.5),
            -F.softplus(-h_t)
        )
        
        # Gate
        gate_sigmoid = torch.sigmoid(g_t)
        log_gate = torch.log(gate_sigmoid.clamp(min=1e-8))
        log_one_minus_gate = torch.log((1 - gate_sigmoid).clamp(min=1e-8))
        
        # Update ALL dimensions together
        h_log = torch.logaddexp(
            log_one_minus_gate + h_log,
            log_gate + log_h_new
        )
        
        # Clamp and output
        h_log = h_log.clamp(min=-20.0, max=20.0)
        outputs.append(torch.exp(h_log))
    
    # Stack outputs and return
    output = torch.stack(outputs, dim=1)  # [B, T, D]
    return output, h_log

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
        
        # TEMPORARY: Use PyTorch implementation until Triton kernel is fixed
        h_output, h_log_final = nau_gru_sequential_forward(
            h, g, prev_hidden, B, T, self.dim_inner
        )
        
        # Project back
        output = self.to_out(h_output)
        
        if return_next_prev_hidden:
            # Return the actual log-space hidden state from kernel
            # CRITICAL: Must be [B, 1, D] to match minGRU's format!
            return output, h_log_final.unsqueeze(1).contiguous()
        return output
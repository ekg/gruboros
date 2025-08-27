"""
Hybrid GRU: PyTorch matmuls + Triton fused cell logic.

Best of both worlds:
- PyTorch's optimized CUBLAS for matrix multiplication
- Triton kernel for fused GRU cell computation
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gru_cell_fused(
    # Gate inputs from matmul
    gates_input_ptr, gates_hidden_ptr,
    # Hidden state
    h_in_ptr, h_out_ptr,
    # Dimensions
    batch_size, hidden_dim,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused GRU cell computation ONLY.
    Assumes gates are already computed by PyTorch matmul.
    
    Input: gates_input[B, 3*H], gates_hidden[B, 3*H], h_in[B, H]
    Output: h_out[B, H]
    """
    # Program for each batch element
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    if pid_batch >= batch_size:
        return
    
    # Process this block of hidden dimensions
    block_start = pid_block * BLOCK_SIZE
    if block_start >= hidden_dim:
        return
        
    # Offsets for this block
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    
    # Load input gates for this block (convert to fp32 for computation)
    base_offset = pid_batch * 3 * hidden_dim
    
    i_r = tl.load(gates_input_ptr + base_offset + offs, mask=mask, other=0.0).to(tl.float32)
    i_z = tl.load(gates_input_ptr + base_offset + hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    i_n = tl.load(gates_input_ptr + base_offset + 2*hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Load hidden gates for this block (convert to fp32 for computation)
    h_r = tl.load(gates_hidden_ptr + base_offset + offs, mask=mask, other=0.0).to(tl.float32)
    h_z = tl.load(gates_hidden_ptr + base_offset + hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    h_n = tl.load(gates_hidden_ptr + base_offset + 2*hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Load previous hidden state (convert to fp32 for computation)
    h_prev_offset = pid_batch * hidden_dim + offs
    h_prev = tl.load(h_in_ptr + h_prev_offset, mask=mask, other=0.0).to(tl.float32)
    
    # GRU cell computation (fused)
    r = tl.sigmoid(i_r + h_r)
    z = tl.sigmoid(i_z + h_z)
    
    # Candidate - tanh implemented manually
    n_pre = i_n + r * h_n
    exp_2x = tl.exp(2.0 * n_pre)
    n = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Update hidden
    h_new = (1.0 - z) * h_prev + z * n
    
    # Store result (convert back to bf16 for storage)
    tl.store(h_out_ptr + h_prev_offset, h_new.to(tl.bfloat16), mask=mask)


class HybridFusedGRU(nn.Module):
    """
    Hybrid GRU using PyTorch matmul + Triton cell fusion.
    
    This should be faster than pure Triton but with fusion benefits.
    """
    
    def __init__(self, dim: int, expansion_factor: float = 1.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        
        # Standard GRU weights
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner)
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        import math
        std = 1.0 / math.sqrt(self.dim_inner)
        for module in [self.input_projection, self.hidden_projection]:
            nn.init.uniform_(module.weight, -std, std)
            nn.init.uniform_(module.bias, -std, std)
        nn.init.normal_(self.to_out.weight, 0.0, 0.02)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        if prev_hidden is None:
            h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h = prev_hidden if prev_hidden.shape[0] == B else torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        
        # Pre-compute ALL input projections at once (FAST!)
        input_gates_all = self.input_projection(x)  # [B, T, 3*H]
        
        outputs = []
        
        # Process timesteps sequentially
        for t in range(T):
            # Get input gates for this timestep
            input_gates = input_gates_all[:, t].contiguous()  # [B, 3*H]
            
            # Compute hidden gates with PyTorch matmul
            hidden_gates = self.hidden_projection(h).contiguous()  # [B, 3*H]
            
            # Prepare output tensor
            h_new = torch.empty_like(h)
            
            # Launch Triton kernel for fused cell computation
            BLOCK_SIZE = min(128, triton.next_power_of_2(self.dim_inner))
            grid = (B, triton.cdiv(self.dim_inner, BLOCK_SIZE))
            
            gru_cell_fused[grid](
                input_gates, hidden_gates,
                h, h_new,
                B, self.dim_inner,
                BLOCK_SIZE
            )
            
            h = h_new
            outputs.append(h)
        
        # Stack outputs and apply final projection
        h_seq = torch.stack(outputs, dim=1)
        out = self.to_out(h_seq) + x  # Residual
        
        if return_next_prev_hidden:
            return out, h
        return out


# Export as NAU_GRU
NAU_GRU = HybridFusedGRU


if __name__ == "__main__":
    print("Testing Hybrid Fused GRU...")
    
    device = 'cuda'
    B, T, D = 8, 64, 256
    
    model = HybridFusedGRU(D).to(device).to(torch.bfloat16)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16) * 0.1
    
    # Test forward
    out = model(x)
    print(f"✓ Forward pass successful!")
    print(f"Output shape: {out.shape}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens = B * T * 10
    print(f"Speed: {tokens/elapsed:.0f} tok/s")
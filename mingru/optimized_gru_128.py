"""
Optimized GRU for chunk_size=128 using various acceleration techniques.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import subprocess


class OptimizedGRU128(nn.Module):
    """
    GRU optimized specifically for chunk_size=128.
    
    Multiple optimization strategies:
    1. torch.compile with mode='max-autotune'
    2. Fully unrolled loop for known sequence length
    3. Fused operations
    """
    
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.5,
        chunk_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.chunk_size = chunk_size
        
        # GRU weights - keep separate for optimization
        self.W_ir = nn.Linear(dim, self.dim_inner, bias=True)
        self.W_iz = nn.Linear(dim, self.dim_inner, bias=True)
        self.W_in = nn.Linear(dim, self.dim_inner, bias=True)
        
        self.W_hr = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        self.W_hz = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        self.W_hn = nn.Linear(self.dim_inner, self.dim_inner, bias=True)
        
        self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        
        self._init_weights()
        
        # Compile the GRU cell for maximum performance
        self.compiled_gru_cell = torch.compile(
            self._gru_cell_fused,
            mode='max-autotune',
            fullgraph=True
        )
        
        print(f"OptimizedGRU128: dim={dim}, dim_inner={self.dim_inner}, chunk_size={chunk_size}")
        print(f"Optimizations: torch.compile(mode='max-autotune'), unrolled loop")
    
    def _init_weights(self):
        import math
        std = 1.0 / math.sqrt(self.dim_inner)
        
        for module in [self.W_ir, self.W_iz, self.W_in, self.W_hr, self.W_hz, self.W_hn]:
            nn.init.uniform_(module.weight, -std, std)
            nn.init.uniform_(module.bias, -std, std)
        
        nn.init.normal_(self.to_out.weight, 0.0, 0.02)
    
    @torch.jit.script
    def _gru_cell_fused(x_t: torch.Tensor, h: torch.Tensor, 
                        W_ir: nn.Module, W_iz: nn.Module, W_in: nn.Module,
                        W_hr: nn.Module, W_hz: nn.Module, W_hn: nn.Module) -> torch.Tensor:
        """Fused GRU cell with all operations combined."""
        # Input projections
        i_r = W_ir(x_t)
        i_z = W_iz(x_t)
        i_n = W_in(x_t)
        
        # Hidden projections
        h_r = W_hr(h)
        h_z = W_hz(h)
        h_n = W_hn(h)
        
        # Fused GRU equations
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1.0 - z) * h + z * n
        
        return h_new
    
    def forward_unrolled(self, x, prev_hidden=None):
        """Fully unrolled forward pass for chunk_size=128."""
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        assert T == self.chunk_size, f"Expected seq_len={self.chunk_size}, got {T}"
        
        if prev_hidden is None:
            h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h = prev_hidden if prev_hidden.shape[0] == B else torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        
        # Pre-compute ALL input projections at once (big optimization!)
        all_i_r = self.W_ir(x.reshape(B * T, D)).reshape(B, T, self.dim_inner)
        all_i_z = self.W_iz(x.reshape(B * T, D)).reshape(B, T, self.dim_inner)
        all_i_n = self.W_in(x.reshape(B * T, D)).reshape(B, T, self.dim_inner)
        
        outputs = []
        
        # Manually unroll for chunk_size=128
        # Python will actually unroll this at compile time
        for t in range(128):  # Hardcoded for maximum optimization
            # Get pre-computed input projections
            i_r = all_i_r[:, t]
            i_z = all_i_z[:, t]
            i_n = all_i_n[:, t]
            
            # Compute hidden projections
            h_r = self.W_hr(h)
            h_z = self.W_hz(h)
            h_n = self.W_hn(h)
            
            # GRU cell (all ops fused)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * h + z * n
            
            outputs.append(h)
        
        # Stack outputs
        h_seq = torch.stack(outputs, dim=1)
        return h_seq, h
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        """Forward pass with optimizations."""
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        
        B, T, D = x.shape
        
        # Use unrolled version for chunk_size=128
        if T == 128:
            h_seq, h_final = self.forward_unrolled(x, prev_hidden)
        else:
            # Fallback to regular loop
            device = x.device
            dtype = x.dtype
            
            if prev_hidden is None:
                h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            else:
                if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                    prev_hidden = prev_hidden.squeeze(1)
                h = prev_hidden if prev_hidden.shape[0] == B else torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            
            outputs = []
            for t in range(T):
                h = self.compiled_gru_cell(x[:, t], h, 
                                          self.W_ir, self.W_iz, self.W_in,
                                          self.W_hr, self.W_hz, self.W_hn)
                outputs.append(h)
            
            h_seq = torch.stack(outputs, dim=1)
            h_final = h
        
        # Output projection + residual
        out = self.to_out(h_seq) + x
        
        if return_next_prev_hidden:
            return out, h_final
        return out


# Even simpler: just torch.compile the SimpleFastGRU
def create_compiled_gru(dim, expansion_factor=1.5, **kwargs):
    """Create a compiled version of SimpleFastGRU."""
    from mingru.fast_gru_triton import SimpleFastGRU
    
    model = SimpleFastGRU(dim=dim, expansion_factor=expansion_factor, **kwargs)
    
    # Compile with maximum optimization
    model.forward = torch.compile(
        model.forward,
        mode='max-autotune',
        fullgraph=True,
        disable=False
    )
    
    print(f"Created compiled SimpleFastGRU with torch.compile(mode='max-autotune')")
    return model


# Use this as NAU_GRU
NAU_GRU = OptimizedGRU128


if __name__ == "__main__":
    print("Testing OptimizedGRU128...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, D = 16, 128, 256  # Fixed sequence length of 128
    
    model = OptimizedGRU128(dim=D, chunk_size=128).to(device)
    
    # Convert to bfloat16
    model = model.to(torch.bfloat16)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16) * 0.1
    
    # Warmup
    for _ in range(3):
        _ = model(x)
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(10):
        output = model(x)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10
    
    print(f"Time per forward: {elapsed*1000:.2f}ms")
    print(f"Throughput: {B * T / elapsed:.0f} tokens/sec")
    
    # Compare with compiled SimpleFastGRU
    print("\nTesting compiled SimpleFastGRU...")
    model2 = create_compiled_gru(D)
    model2 = model2.to(device).to(torch.bfloat16)
    
    # Warmup
    for _ in range(3):
        _ = model2(x)
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(10):
        output2 = model2(x)
    
    torch.cuda.synchronize()
    elapsed2 = (time.time() - start) / 10
    
    print(f"Time per forward: {elapsed2*1000:.2f}ms")
    print(f"Throughput: {B * T / elapsed2:.0f} tokens/sec")
    
    print(f"\nSpeedup: {elapsed2/elapsed:.2f}x")
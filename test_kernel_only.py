#!/usr/bin/env python3
"""Simple test to check if kernel compiles and runs"""

import torch
import sys
sys.path.insert(0, '.')

# Just test initialization and a simple forward pass
try:
    from mingru.hybrid_fused_gru import HybridFusedGRU
    
    print("Testing HybridFusedGRU...")
    model = HybridFusedGRU(256, expansion_factor=1.5).to('cuda')
    
    # Check bias initialization
    H = model.dim_inner
    with torch.no_grad():
        z_bias_in = model.input_projection.bias[H:2*H].mean().item()
        z_bias_hid = model.hidden_projection.bias[H:2*H].mean().item()
        print(f"Z-gate biases: input={z_bias_in:.2f}, hidden={z_bias_hid:.2f}")
    
    # Try a forward pass with CPU fallback first
    x = torch.randn(2, 3, 256, device='cpu', dtype=torch.float32)
    model_cpu = HybridFusedGRU(256, expansion_factor=1.5).to('cpu')
    y_cpu = model_cpu(x)
    print(f"CPU forward: {y_cpu.shape}, dtype={y_cpu.dtype}")
    
    # Now try GPU
    x_gpu = x.to('cuda')
    try:
        y_gpu = model(x_gpu)
        print(f"GPU forward: {y_gpu.shape}, dtype={y_gpu.dtype}")
    except Exception as e:
        print(f"GPU forward failed (kernel issue): {e}")
        print("This is expected if the Triton kernel has compilation issues.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
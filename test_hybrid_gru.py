#!/usr/bin/env python3
"""
Unit test for HybridFusedGRU to verify:
1. Dtype consistency (no fp32/bf16 mismatch)
2. Gradient flow works correctly
3. Z-gate bias initialization works as expected
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mingru.hybrid_fused_gru import HybridFusedGRU

def test_dtype_consistency():
    """Test that HybridFusedGRU handles bf16 correctly with Triton kernel."""
    print("Testing dtype consistency...")
    
    # Test dimensions
    B, T, D = 4, 8, 512
    H = int(D * 1.5)  # expansion factor 1.5
    
    # Test both bf16 and fp32
    for dtype in [torch.bfloat16, torch.float32]:
        dtype_name = "bf16" if dtype == torch.bfloat16 else "fp32"
        print(f"\n  Testing {dtype_name}...")
        
        # Create model
        model = HybridFusedGRU(
            dim=D, 
            expansion_factor=1.5,
            z_bias_input=-1.0,
            z_bias_hidden=0.0
        ).cuda().to(dtype)
        
        # Create input
        x = torch.randn(B, T, D, device='cuda', dtype=dtype)
        
        # Test single-token path (T=1)
        x_single = x[:, :1, :]
        with torch.no_grad():
            out_single, h_new = model(x_single, return_next_prev_hidden=True)
        
        assert out_single.dtype == dtype, f"Single-token output dtype mismatch: {out_single.dtype} != {dtype}"
        assert h_new.dtype == dtype, f"Single-token hidden dtype mismatch: {h_new.dtype} != {dtype}"
        assert not torch.isnan(out_single).any(), "NaN in single-token output"
        assert not torch.isnan(h_new).any(), "NaN in single-token hidden state"
        print(f"    ✓ Single-token path: dtype={out_single.dtype}, no NaNs")
        
        # Test multi-token path
        with torch.no_grad():
            out_multi, h_final = model(x, return_next_prev_hidden=True)
        
        assert out_multi.dtype == dtype, f"Multi-token output dtype mismatch: {out_multi.dtype} != {dtype}"
        assert h_final.dtype == dtype, f"Multi-token hidden dtype mismatch: {h_final.dtype} != {dtype}"
        assert not torch.isnan(out_multi).any(), "NaN in multi-token output"
        assert not torch.isnan(h_final).any(), "NaN in multi-token hidden state"
        print(f"    ✓ Multi-token path: dtype={out_multi.dtype}, no NaNs")
        
        # Test gradient flow (only in fp32 for simplicity)
        if dtype == torch.float32:
            x.requires_grad = True
            out, _ = model(x)
            loss = out.mean()
            loss.backward()
            
            assert x.grad is not None, "No gradient on input"
            assert not torch.isnan(x.grad).any(), "NaN in gradients"
            print(f"    ✓ Gradient flow works: grad norm = {x.grad.norm().item():.4f}")


def test_z_gate_initialization():
    """Test that z-gate biases are initialized correctly."""
    print("\nTesting z-gate initialization...")
    
    D = 256
    test_configs = [
        {"z_bias_input": -1.0, "z_bias_hidden": 0.0},
        {"z_bias_input": -2.0, "z_bias_hidden": -2.0},
        {"z_bias_input": -3.0, "z_bias_hidden": -1.0},
    ]
    
    for config in test_configs:
        model = HybridFusedGRU(dim=D, expansion_factor=1.5, **config)
        
        # Check input projection z-gate bias
        H = model.dim_inner
        input_z_bias = model.input_projection.bias[H:2*H].mean().item()
        hidden_z_bias = model.hidden_projection.bias[H:2*H].mean().item()
        
        # Allow small floating point differences
        assert abs(input_z_bias - config["z_bias_input"]) < 1e-5, \
            f"Input z-bias mismatch: {input_z_bias} != {config['z_bias_input']}"
        assert abs(hidden_z_bias - config["z_bias_hidden"]) < 1e-5, \
            f"Hidden z-bias mismatch: {hidden_z_bias} != {config['z_bias_hidden']}"
        
        print(f"  ✓ Config {config}: input_z={input_z_bias:.2f}, hidden_z={hidden_z_bias:.2f}")


def test_performance_sanity():
    """Quick performance sanity check - bf16 should be faster than fp32."""
    print("\nTesting performance (relative)...")
    
    import time
    
    B, T, D = 32, 256, 768
    
    times = {}
    for dtype in [torch.float32, torch.bfloat16]:
        dtype_name = "fp32" if dtype == torch.float32 else "bf16"
        
        model = HybridFusedGRU(dim=D, expansion_factor=1.5).cuda().to(dtype)
        x = torch.randn(B, T, D, device='cuda', dtype=dtype)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        
        # Time
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        times[dtype_name] = elapsed
        print(f"  {dtype_name}: {elapsed:.4f}s for 10 forward passes")
    
    speedup = times["fp32"] / times["bf16"]
    print(f"  ✓ BF16 speedup: {speedup:.2f}x")
    if speedup < 1.0:
        print("  ⚠️  Warning: BF16 is slower than FP32 (might be due to small model size)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)
    
    print("=" * 60)
    print("HybridFusedGRU Unit Tests")
    print("=" * 60)
    
    try:
        test_dtype_consistency()
        test_z_gate_initialization()
        test_performance_sanity()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
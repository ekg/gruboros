#!/usr/bin/env python3
"""
Quick validation tests for HybridFusedGRU changes:
1. Test kernel forward pass
2. Verify z-gate bias initialization
"""

import torch
import sys
sys.path.insert(0, '.')
from mingru.hybrid_fused_gru import HybridFusedGRU

def test_kernel_forward():
    """Test that the kernel runs without errors"""
    print("Testing kernel forward pass...")
    try:
        model = HybridFusedGRU(256).to('cuda')
        x = torch.randn(2, 3, 256, device='cuda', dtype=torch.float32)
        y = model(x)
        print(f"✓ Kernel forward pass successful: output shape {y.shape}")
        return True
    except Exception as e:
        print(f"✗ Kernel forward pass failed: {e}")
        return False

def test_z_gate_bias():
    """Verify z-gate bias is initialized to -2.0"""
    print("\nTesting z-gate bias initialization...")
    try:
        model = HybridFusedGRU(256).to('cuda')
        H = model.dim_inner
        
        # Check input projection z-gate bias
        with torch.no_grad():
            z_bias_in = model.input_projection.bias[H:2*H].mean().item()
            z_bias_hid = model.hidden_projection.bias[H:2*H].mean().item()
            
            # Check if biases are close to -2.0
            tolerance = 0.01
            input_ok = abs(z_bias_in - (-2.0)) < tolerance
            hidden_ok = abs(z_bias_hid - (-2.0)) < tolerance
            
            print(f"  Input projection z-bias: {z_bias_in:.4f} {'✓' if input_ok else '✗'}")
            print(f"  Hidden projection z-bias: {z_bias_hid:.4f} {'✓' if hidden_ok else '✗'}")
            
            # Also check that residual projection is zero-initialized (if it exists)
            if not isinstance(model.to_out, torch.nn.Identity):
                residual_mean = model.to_out.weight.abs().mean().item()
                residual_ok = residual_mean < 0.01
                print(f"  Residual projection weight mean: {residual_mean:.6f} {'✓' if residual_ok else '✗'}")
            else:
                print("  Residual projection: Identity (no expansion) ✓")
                residual_ok = True
            
            return input_ok and hidden_ok and residual_ok
    except Exception as e:
        print(f"✗ Bias verification failed: {e}")
        return False

def test_fp32_processing():
    """Verify that the model processes in fp32"""
    print("\nTesting fp32 processing...")
    try:
        model = HybridFusedGRU(256).to('cuda')
        x = torch.randn(2, 3, 256, device='cuda', dtype=torch.float32)
        
        # Forward pass
        y = model(x)
        
        # Check that output is fp32
        dtype_ok = y.dtype == torch.float32
        print(f"  Output dtype: {y.dtype} {'✓' if dtype_ok else '✗'}")
        
        # Check intermediate hidden state during single-token generation
        x_single = torch.randn(2, 1, 256, device='cuda', dtype=torch.float32)
        y_single, h_new = model(x_single, return_next_prev_hidden=True)
        
        hidden_dtype_ok = h_new.dtype == torch.float32
        print(f"  Hidden state dtype: {h_new.dtype} {'✓' if hidden_dtype_ok else '✗'}")
        
        return dtype_ok and hidden_dtype_ok
    except Exception as e:
        print(f"✗ FP32 processing test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("HybridFusedGRU Validation Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. These tests require a GPU.")
        sys.exit(1)
    
    results = []
    results.append(test_kernel_forward())
    results.append(test_z_gate_bias())
    results.append(test_fp32_processing())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please review the output above.")
        sys.exit(1)
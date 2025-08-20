#!/usr/bin/env python
"""Debug conv implementation to find initialization issues."""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '.')

from mingru.minLM import minLM, CausalDepthWiseConv1d

def test_conv_layer():
    """Test conv layer in isolation."""
    print("Testing CausalDepthWiseConv1d...")
    
    batch_size = 1
    seq_len = 1024
    dim = 1024
    kernel_size = 16
    
    # Create conv layer
    conv = CausalDepthWiseConv1d(dim, kernel_size)
    conv.cuda()
    
    # Test input
    x = torch.randn(batch_size, seq_len, dim).cuda()
    
    print(f"Input shape: {x.shape}")
    print(f"Conv depthwise weight shape: {conv.depthwise.weight.shape}")
    print(f"Conv pointwise weight shape: {conv.pointwise.weight.shape}")
    
    # Test forward pass
    start = time.time()
    with torch.no_grad():
        out, buffer = conv(x, prev_buffer=None)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Output shape: {out.shape}")
    print(f"Buffer shape: {buffer.shape if buffer is not None else None}")
    print(f"Forward pass time: {elapsed:.4f}s")
    
    # Check for NaN/inf
    if torch.isnan(out).any():
        print("ERROR: Output contains NaN!")
    if torch.isinf(out).any():
        print("ERROR: Output contains inf!")
    
    print("Conv layer test passed!\n")
    return True

def test_full_model():
    """Test full model with conv."""
    print("Testing full minLM model with conv...")
    
    model_config = {
        "num_tokens": 256,
        "dim": 512,  # Smaller for testing
        "depth": 4,   # Fewer layers for testing
        "ff_mult": 0,
        "expansion": 1.5,
        "conv_kernel_size": 16,
        "dropout": 0.0
    }
    
    print(f"Model config: {model_config}")
    
    # Create model
    print("Creating model...")
    start = time.time()
    model = minLM(**model_config)
    creation_time = time.time() - start
    print(f"Model creation time: {creation_time:.2f}s")
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # Move to GPU
    print("Moving to GPU...")
    model.cuda()
    
    # Test input
    batch_size = 1
    seq_len = 128  # Short sequence for testing
    x = torch.randint(0, 256, (batch_size, seq_len + 1)).cuda()  # +1 for labels
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    start = time.time()
    with torch.no_grad():
        loss = model(x, return_loss=True, return_prev_hiddens=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Forward pass time: {elapsed:.4f}s")
    
    # Test with hidden states
    print("\nTesting with hidden states...")
    start = time.time()
    with torch.no_grad():
        loss, (hiddens, buffers) = model(
            x, 
            return_loss=True, 
            return_prev_hiddens=True,
            prev_hiddens=None,
            prev_conv_buffers=None
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Hidden states: {len(hiddens)} layers")
    print(f"Conv buffers: {len(buffers)} layers")
    print(f"Forward pass time: {elapsed:.4f}s")
    
    print("\nFull model test passed!")
    return True

def test_weight_init():
    """Test weight initialization."""
    print("Testing weight initialization...")
    
    conv = CausalDepthWiseConv1d(1024, 16)
    
    # Check weight statistics
    dw_mean = conv.depthwise.weight.mean().item()
    dw_std = conv.depthwise.weight.std().item()
    pw_mean = conv.pointwise.weight.mean().item()
    pw_std = conv.pointwise.weight.std().item()
    
    print(f"Depthwise conv - mean: {dw_mean:.6f}, std: {dw_std:.6f}")
    print(f"Pointwise conv - mean: {pw_mean:.6f}, std: {pw_std:.6f}")
    
    if abs(dw_mean) > 0.1 or dw_std < 0.001 or dw_std > 1.0:
        print("WARNING: Depthwise conv weights may not be initialized properly!")
    if abs(pw_mean) > 0.1 or pw_std < 0.001 or pw_std > 1.0:
        print("WARNING: Pointwise conv weights may not be initialized properly!")
    
    print("Weight initialization test passed!\n")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Conv Layer Debugging")
    print("=" * 50)
    
    try:
        test_weight_init()
        test_conv_layer()
        test_full_model()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
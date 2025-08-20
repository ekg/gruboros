#!/usr/bin/env python3
"""Simple test to debug conv initialization issue."""

import sys
import torch
import torch.nn as nn

print("Testing conv initialization...")

# Test parameters
batch_size = 2
seq_len = 128
dim = 512
kernel_size = 16

print(f"Parameters: batch={batch_size}, seq={seq_len}, dim={dim}, kernel={kernel_size}")

# Test basic conv layers
print("\n1. Testing depthwise conv creation...")
try:
    depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, bias=False, padding=0)
    print(f"   Depthwise conv created: {depthwise}")
    print(f"   Weight shape: {depthwise.weight.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n2. Testing pointwise conv creation...")
try:
    pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
    print(f"   Pointwise conv created: {pointwise}")
    print(f"   Weight shape: {pointwise.weight.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n3. Testing forward pass...")
try:
    x = torch.randn(batch_size, seq_len, dim)
    print(f"   Input shape: {x.shape}")
    
    # Transpose for conv
    x_conv = x.transpose(1, 2).contiguous()
    print(f"   After transpose: {x_conv.shape}")
    
    # Pad
    x_padded = torch.nn.functional.pad(x_conv, (kernel_size-1, 0), value=0.)
    print(f"   After padding: {x_padded.shape}")
    
    # Apply convolutions
    out = depthwise(x_padded)
    print(f"   After depthwise: {out.shape}")
    
    out = pointwise(out)
    print(f"   After pointwise: {out.shape}")
    
    # Transpose back
    out = out.transpose(1, 2).contiguous()
    print(f"   Final output: {out.shape}")
    
    print("\n   Forward pass successful!")
except Exception as e:
    print(f"   ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing with CUDA if available...")
if torch.cuda.is_available():
    try:
        device = torch.device('cuda:0')
        print(f"   Moving to device: {device}")
        
        depthwise = depthwise.to(device)
        pointwise = pointwise.to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        x_conv = x.transpose(1, 2).contiguous()
        x_padded = torch.nn.functional.pad(x_conv, (kernel_size-1, 0), value=0.)
        out = depthwise(x_padded)
        out = pointwise(out)
        out = out.transpose(1, 2).contiguous()
        
        print(f"   CUDA forward pass successful! Output shape: {out.shape}")
    except Exception as e:
        print(f"   ERROR with CUDA: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   CUDA not available")

print("\n5. Testing CausalDepthWiseConv1d class...")
try:
    from mingru.minLM import CausalDepthWiseConv1d
    
    conv_layer = CausalDepthWiseConv1d(dim, kernel_size)
    print(f"   Created CausalDepthWiseConv1d: {conv_layer}")
    
    x = torch.randn(batch_size, seq_len, dim)
    out, buffer = conv_layer(x, prev_buffer=None)
    print(f"   Forward pass successful!")
    print(f"   Output shape: {out.shape}")
    print(f"   Buffer shape: {buffer.shape}")
    
    if torch.cuda.is_available():
        conv_layer = conv_layer.to('cuda')
        x = x.to('cuda')
        out, buffer = conv_layer(x, prev_buffer=None)
        print(f"   CUDA forward pass successful!")
        
except ImportError as e:
    print(f"   Could not import CausalDepthWiseConv1d: {e}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
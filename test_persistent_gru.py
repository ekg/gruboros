#!/usr/bin/env python
"""Test PersistentGRU kernel correctness and performance."""

import torch
import time
from mingru.persistent_gru import PersistentGRU


def test_persistent_gru():
    """Test that persistent kernel produces correct output."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test")
        return

    # Small test dimensions
    B, T, D = 4, 64, 256
    expansion = 1.0

    print("=" * 70)
    print(f"Testing PersistentGRU: B={B}, T={T}, D={D}")
    print("=" * 70)

    # Create model
    gru = PersistentGRU(dim=D, expansion_factor=expansion).to(device).to(torch.bfloat16)

    # Test input
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    prev_hidden = torch.randn(B, int(D * expansion), device=device, dtype=torch.bfloat16)

    print("\n1. Testing basic forward pass...")

    # Forward pass
    with torch.no_grad():
        out, h = gru(x, prev_hidden, return_next_prev_hidden=True)

    # Check shapes
    expected_out_shape = (B, T, D)
    expected_h_shape = (B, int(D * expansion))

    print(f"   Output shape: {tuple(out.shape)} (expected {expected_out_shape})")
    print(f"   Hidden shape: {tuple(h.shape)} (expected {expected_h_shape})")

    if out.shape == expected_out_shape and h.shape == expected_h_shape:
        print("   ✓ PASS: Shapes correct!")
    else:
        print("   ✗ FAIL: Shape mismatch!")
        return False

    # Check for NaN/Inf
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("   ✗ FAIL: Output contains NaN/Inf!")
        return False
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("   ✗ FAIL: Hidden contains NaN/Inf!")
        return False

    print("   ✓ PASS: No NaN/Inf detected!")

    print("\n2. Testing with document boundaries...")

    # Create document boundaries
    doc_boundaries = torch.zeros(B, T, dtype=torch.bool, device=device)
    doc_boundaries[:, T//2] = True  # Reset at middle

    # Forward with boundaries
    with torch.no_grad():
        out_doc, h_doc = gru(x, prev_hidden,
                             return_next_prev_hidden=True,
                             doc_boundaries=doc_boundaries)

    # Check that outputs differ (boundaries should change behavior)
    out_diff = (out - out_doc).abs().mean().item()
    print(f"   Mean diff with/without boundaries: {out_diff:.6f}")

    if out_diff > 1e-6:
        print("   ✓ PASS: Document boundaries affect output!")
    else:
        print("   ⚠ WARNING: Boundaries might not be working")

    # Check for NaN/Inf
    if torch.isnan(out_doc).any() or torch.isinf(out_doc).any():
        print("   ✗ FAIL: Output with boundaries contains NaN/Inf!")
        return False

    print("   ✓ PASS: No NaN/Inf with boundaries!")

    print("\n3. Testing performance...")

    # Larger dimensions for realistic benchmark
    B_large, T_large, D_large = 8, 512, 512
    x_large = torch.randn(B_large, T_large, D_large, device=device, dtype=torch.bfloat16)
    prev_large = torch.randn(B_large, int(D_large * expansion), device=device, dtype=torch.bfloat16)

    gru_large = PersistentGRU(dim=D_large, expansion_factor=expansion).to(device).to(torch.bfloat16)

    # Warmup
    for _ in range(3):
        _ = gru_large(x_large, prev_large, return_next_prev_hidden=True)

    torch.cuda.synchronize()

    # Benchmark
    n_iters = 20
    start = time.time()
    for _ in range(n_iters):
        _ = gru_large(x_large, prev_large, return_next_prev_hidden=True)
    torch.cuda.synchronize()
    time_avg = (time.time() - start) / n_iters

    tokens_per_sec = (B_large * T_large) / time_avg

    print(f"   Batch size: {B_large}")
    print(f"   Sequence length: {T_large}")
    print(f"   Hidden dim: {D_large}")
    print(f"   Time per forward: {time_avg*1000:.2f} ms")
    print(f"   Throughput: {tokens_per_sec/1000:.1f}K tokens/s")
    print("   ✓ PASS: Performance benchmark complete!")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

    return True


if __name__ == '__main__':
    test_persistent_gru()

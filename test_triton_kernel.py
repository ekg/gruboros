#!/usr/bin/env python3
"""
Test script to verify Triton fused matmul+perturbation kernel correctness.
"""

import torch
import time
from zero_order_triton import matmul_with_perturbation

def test_kernel_correctness():
    """Test that Triton kernel produces same results as PyTorch"""
    print("="*80)
    print("Testing Triton Kernel Correctness")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test dimensions
    M, K, N = 128, 256, 512
    epsilon = 0.01

    # Create test tensors
    x = torch.randn(M, K, device=device)
    w = torch.randn(K, N, device=device)
    pert = torch.randn(K, N, device=device)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  w: {w.shape}")
    print(f"  pert: {pert.shape}")

    # Reference implementation (PyTorch)
    start = time.time()
    w_perturbed_ref = w + epsilon * pert
    y_ref = torch.matmul(x, w_perturbed_ref)
    time_ref = time.time() - start

    print(f"\nPyTorch reference:")
    print(f"  Output shape: {y_ref.shape}")
    print(f"  Time: {time_ref*1000:.3f}ms")
    print(f"  Sample values: {y_ref[0, :5]}")

    # Triton implementation
    start = time.time()
    y_triton = matmul_with_perturbation(x, w, pert, epsilon)
    time_triton = time.time() - start

    print(f"\nTriton fused kernel:")
    print(f"  Output shape: {y_triton.shape}")
    print(f"  Time: {time_triton*1000:.3f}ms")
    print(f"  Sample values: {y_triton[0, :5]}")

    # Compare results
    max_diff = torch.max(torch.abs(y_ref - y_triton)).item()
    mean_diff = torch.mean(torch.abs(y_ref - y_triton)).item()
    rel_error = mean_diff / torch.mean(torch.abs(y_ref)).item()

    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")

    # Check if results match
    tolerance = 1e-4
    matches = torch.allclose(y_ref, y_triton, rtol=tolerance, atol=tolerance)

    if matches:
        print(f"\n✓ PASSED: Results match within tolerance ({tolerance:.0e})")
        speedup = time_ref / time_triton
        print(f"  Speedup: {speedup:.2f}×")
    else:
        print(f"\n✗ FAILED: Results do not match!")
        print(f"  Max diff {max_diff:.2e} exceeds tolerance {tolerance:.0e}")

    print("="*80)
    return matches


def benchmark_kernel():
    """Benchmark Triton kernel vs PyTorch"""
    print("\n" + "="*80)
    print("Benchmarking Triton Kernel")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test different sizes
    sizes = [
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
    ]

    epsilon = 0.01
    n_trials = 100

    print(f"\nRunning {n_trials} trials per size...\n")

    for M, K, N in sizes:
        x = torch.randn(M, K, device=device)
        w = torch.randn(K, N, device=device)
        pert = torch.randn(K, N, device=device)

        # Warmup
        for _ in range(10):
            _ = torch.matmul(x, w + epsilon * pert)
            _ = matmul_with_perturbation(x, w, pert, epsilon)

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_trials):
            _ = torch.matmul(x, w + epsilon * pert)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / n_trials

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_trials):
            _ = matmul_with_perturbation(x, w, pert, epsilon)
        torch.cuda.synchronize()
        time_triton = (time.time() - start) / n_trials

        speedup = time_pytorch / time_triton

        print(f"Size [{M:4d} x {K:4d}] @ [{K:4d} x {N:4d}]:")
        print(f"  PyTorch: {time_pytorch*1000:6.3f}ms")
        print(f"  Triton:  {time_triton*1000:6.3f}ms")
        print(f"  Speedup: {speedup:5.2f}×")
        print()

    print("="*80)


if __name__ == '__main__':
    # Test correctness
    success = test_kernel_correctness()

    if success:
        # Benchmark if correctness passed
        benchmark_kernel()
    else:
        print("\nSkipping benchmark due to correctness failure")

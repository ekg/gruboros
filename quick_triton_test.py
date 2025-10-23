#!/usr/bin/env python3
"""Quick Triton kernel correctness test for fast iteration"""

import torch
from zero_order_triton import matmul_with_perturbation

def test_small():
    """Test on tiny matrices for fast iteration"""
    device = torch.device('cuda')

    # Very small test
    M, K, N = 8, 16, 32
    epsilon = 0.1

    x = torch.randn(M, K, device=device)
    w = torch.randn(K, N, device=device)
    pert = torch.randn(K, N, device=device)

    # Reference
    y_ref = torch.matmul(x, w + epsilon * pert)

    # Triton
    y_triton = matmul_with_perturbation(x, w, pert, epsilon)

    # Compare
    max_diff = torch.max(torch.abs(y_ref - y_triton)).item()
    rel_error = (torch.mean(torch.abs(y_ref - y_triton)) / torch.mean(torch.abs(y_ref))).item()

    passed = torch.allclose(y_ref, y_triton, rtol=1e-4, atol=1e-4)

    print(f"Small test [{M}x{K}] @ [{K}x{N}]:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Rel error: {rel_error:.2e}")
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed

if __name__ == '__main__':
    test_small()

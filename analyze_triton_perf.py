#!/usr/bin/env python3
"""Analyze Triton kernel performance issues"""

import torch
from zero_order_triton import matmul_with_perturbation

# Test case from failing test
M, K, N = 128, 256, 512
device = torch.device('cuda')

x = torch.randn(M, K, device=device)
w = torch.randn(K, N, device=device)
pert = torch.randn(K, N, device=device)
epsilon = 0.01

print("=" * 80)
print("Performance Analysis")
print("=" * 80)
print(f"\nMatrix dimensions: [{M}, {K}] @ [{K}, {N}]")
print(f"Block sizes will be:")
import triton
BLOCK_M = max(16, min(64, triton.next_power_of_2(M)))
BLOCK_N = max(16, min(64, triton.next_power_of_2(N)))
BLOCK_K = max(16, min(32, triton.next_power_of_2(K)))
print(f"  BLOCK_M = {BLOCK_M}")
print(f"  BLOCK_N = {BLOCK_N}")
print(f"  BLOCK_K = {BLOCK_K}")

grid_m = triton.cdiv(M, BLOCK_M)
grid_n = triton.cdiv(N, BLOCK_N)
num_k_blocks = triton.cdiv(K, BLOCK_K)

print(f"\nGrid dimensions: ({grid_m}, {grid_n})")
print(f"Total thread blocks: {grid_m * grid_n}")
print(f"K-loop iterations per block: {num_k_blocks}")
print(f"Total kernel work: {grid_m * grid_n * num_k_blocks} block-iterations")

# Estimate shared memory usage
elements_per_block = (BLOCK_M * BLOCK_K +  # X block
                     BLOCK_K * BLOCK_N +   # W block
                     BLOCK_K * BLOCK_N +   # P block
                     BLOCK_M * BLOCK_N)    # accumulator
shared_mem_bytes = elements_per_block * 4  # float32
print(f"\nShared memory per block: {shared_mem_bytes:,} bytes ({shared_mem_bytes/1024:.1f} KB)")

print(f"\nIssues detected:")
if grid_m * grid_n > 1024:
    print(f"  ⚠ Very small blocks → {grid_m * grid_n} thread blocks (launch overhead!)")
else:
    print(f"  ✓ Reasonable number of thread blocks")

if num_k_blocks > 16:
    print(f"  ⚠ Many K-iterations: {num_k_blocks} (could be optimized)")
elif num_k_blocks > 8:
    print(f"  ~ Moderate K-iterations: {num_k_blocks}")
else:
    print(f"  ✓ Few K-iterations: {num_k_blocks}")

if shared_mem_bytes > 65536:
    print(f"  ⚠ High shared memory usage: {shared_mem_bytes/1024:.1f} KB")
else:
    print(f"  ✓ Reasonable shared memory: {shared_mem_bytes/1024:.1f} KB")

print(f"\nRecommendations:")
print(f"  - Current BLOCK sizes: {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
print(f"  - PyTorch uses highly optimized cuBLAS")
print(f"  - For THIS SIZE matrix, PyTorch may be faster")
print(f"  - Triton shines for LARGER matrices or when fusion saves memory bandwidth")
print("=" * 80)

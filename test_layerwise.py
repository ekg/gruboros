#!/usr/bin/env python3
"""
Test layer-wise materialized perturbations optimizer.
"""

import torch
import sys
import time

sys.path.insert(0, 'mingru')
from minLM import minLM
from zero_order_layerwise import LayerwiseZeroOrderOptimizer
from zero_order_true_virtual import TrueVirtualZeroOrderOptimizer

def load_tiny_model():
    """Load tiny model for testing"""
    config = {
        'num_tokens': 100277,
        'dim': 384,
        'depth': 2,
        'expansion': 1.0,
        'conv_kernel_size': 4,
        'dropout': 0.0,
        'use_hybrid_gru': True,
        'ff_mult': 0.0,
    }
    model = minLM(**config).cuda()
    model.eval()
    return model

print("="*80)
print("LAYER-WISE MATERIALIZATION TEST")
print("="*80)

# Create test batch
batch = torch.randint(0, 100277, (4, 128), device='cuda')
n_perts = 8

# Test 1: Serial Virtual (baseline)
print("\n1. BASELINE: Serial Virtual Perturbations")
print("-"*40)
model1 = load_tiny_model()
opt1 = TrueVirtualZeroOrderOptimizer(
    model1,
    learning_rate=1e-4,
    epsilon=1e-4,
    n_perturbations=n_perts,
    base_seed=42,
    rank=0
)

# Warmup
_ = opt1.step(None, batch)

# Benchmark
t_start = time.time()
result1 = opt1.step(None, batch)
t_serial = time.time() - t_start

print(f"\n  Time: {t_serial:.2f}s")
print(f"  Loss: {result1['loss']:.4f}")

# Test 2: Layer-wise Materialization
print("\n2. NEW: Layer-Wise Materialized Perturbations")
print("-"*40)
model2 = load_tiny_model()
model2.load_state_dict(model1.state_dict())  # Ensure identical starting point

opt2 = LayerwiseZeroOrderOptimizer(
    model2,
    learning_rate=1e-4,
    epsilon=1e-4,
    n_perturbations=n_perts,
    base_seed=42,
    rank=0
)

# Warmup
_ = opt2.step(None, batch)

# Benchmark
t_start = time.time()
result2 = opt2.step(None, batch)
t_layerwise = time.time() - t_start

print(f"\n  Time: {t_layerwise:.2f}s")
print(f"  Loss: {result2['loss']:.4f}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

speedup = t_serial / t_layerwise

print(f"\nSerial:     {t_serial:.2f}s")
print(f"Layer-wise: {t_layerwise:.2f}s")
print(f"Speedup:    {speedup:.2f}×")

if speedup > 1.5:
    print(f"\n✓ SIGNIFICANT SPEEDUP! Layer-wise is {speedup:.1f}× faster!")
elif speedup > 1.0:
    print(f"\n~ Modest speedup ({speedup:.1f}×)")
elif speedup > 0.8:
    print(f"\n~ Comparable performance (within 20%)")
else:
    print(f"\n✗ Layer-wise is slower ({1/speedup:.1f}× slower)")

# Check correctness
print(f"\nLoss comparison:")
print(f"  Serial:     {result1['loss']:.6f}")
print(f"  Layer-wise: {result2['loss']:.6f}")
print(f"  Difference: {abs(result1['loss'] - result2['loss']):.2e}")

# Compare final weights
max_diff = 0.0
for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    diff = (p1 - p2).abs().max().item()
    if diff > max_diff:
        max_diff = diff

print(f"\nMax weight difference: {max_diff:.2e}")
if max_diff < 1e-3:
    print("✓ Results are IDENTICAL (within tolerance)")
else:
    print("✗ Results DIFFER!")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Layer-wise materialization: {speedup:.2f}× {'faster' if speedup > 1 else 'slower'}")
print(f"Correctness: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

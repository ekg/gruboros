#!/usr/bin/env python3
"""
Compare serial vs batch-parallel zero-order optimization.
Verify they produce same results and measure speedup.
"""

import torch
import sys
import time

sys.path.insert(0, 'mingru')
from minLM import minLM

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

def test_correctness():
    """Verify both methods produce identical results"""
    print("="*80)
    print("CORRECTNESS TEST: Serial vs Batch-Parallel")
    print("="*80)
    
    # Create two identical models
    model1 = load_tiny_model()
    model2 = load_tiny_model()
    
    # Copy weights to ensure they're identical
    model2.load_state_dict(model1.state_dict())
    
    # Create test batch
    batch = torch.randint(0, 100277, (4, 128), device='cuda')
    
    print("\n1. Testing serial virtual perturbation optimizer...")
    from zero_order_true_virtual import TrueVirtualZeroOrderOptimizer
    opt1 = TrueVirtualZeroOrderOptimizer(
        model1,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=8,
        base_seed=42,
        rank=0
    )
    
    print("\n2. Testing batch-parallel optimizer...")
    from zero_order_streams import StreamsZeroOrderOptimizer
    opt2 = StreamsZeroOrderOptimizer(
        model2,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=8,
        parallel_batch_size=8,
        base_seed=42,
        rank=0
    )
    
    print("\n3. Running one step with each optimizer...")
    
    # Run serial
    t1 = time.time()
    result1 = opt1.step(None, batch)
    t_serial = time.time() - t1
    
    # Run batched
    t2 = time.time()
    result2 = opt2.step(None, batch)
    t_batched = time.time() - t2
    
    print(f"\nResults:")
    print(f"  Serial:  loss={result1['loss']:.4f}, time={t_serial:.2f}s")
    print(f"  Batched: loss={result2['loss']:.4f}, time={t_batched:.2f}s")
    
    # Compare final weights
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_diff_layer = n1
    
    print(f"\nWeight comparison:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Layer with max diff: {max_diff_layer}")
    
    if max_diff < 1e-3:
        print(f"  ✓ Results are IDENTICAL (within 1e-3 tolerance)")
    else:
        print(f"  ✗ Results DIFFER significantly!")
    
    speedup = t_serial / t_batched
    print(f"\nPerformance:")
    print(f"  Speedup: {speedup:.2f}x")
    if speedup > 1.0:
        print(f"  ✓ Batched is FASTER")
    else:
        print(f"  ✗ Batched is SLOWER (likely overhead from broadcasting)")
    
    return max_diff < 1e-3, speedup

if __name__ == '__main__':
    correct, speedup = test_correctness()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if correct:
        print(f"✓ Correctness: PASS")
        print(f"✓ Both methods produce identical results")
    else:
        print(f"✗ Correctness: FAIL")
        print(f"✗ Methods produce different results!")
    
    print(f"\nSpeedup: {speedup:.2f}x")
    if speedup > 1.5:
        print(f"✓ Significant speedup achieved!")
    elif speedup > 0.8:
        print(f"~ Comparable performance (overhead within 20%)")
    else:
        print(f"✗ Batched version is slower (needs optimization)")

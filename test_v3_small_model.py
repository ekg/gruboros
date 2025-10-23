#!/usr/bin/env python3
"""
Test V3 optimizer with smaller model to debug vmap + Triton kernel issue

Goal: Get parallel perturbation processing working with HybridGRU
"""

import torch
import torch.nn as nn
import sys
import time

from zero_order_triton_v3 import TritonZeroOrderOptimizerV3


def load_small_model(depth=6, dim=768):
    """Load a smaller model for faster testing (~50M params)"""
    sys.path.insert(0, 'mingru')
    from minLM import minLM

    config = {
        'num_tokens': 100277,
        'dim': dim,  # Smaller
        'depth': depth,  # Fewer layers
        'expansion': 1.0,
        'conv_kernel_size': 4,
        'dropout': 0.0,
        'use_hybrid_gru': True,  # Use Triton kernel
        'ff_mult': 0.0,
    }

    model = minLM(**config).cuda()
    model.eval()

    return model


def test_v3_optimizer(model, pert_batch_size, total_perturbations=32,
                     num_steps=2, data_batch_size=2, seq_len=256):
    """Test V3 with specific parallel batch size"""
    print(f"\n{'='*80}")
    print(f"TESTING: V3 Optimizer with pert_batch_size = {pert_batch_size}")
    print(f"  (processing {pert_batch_size} perturbations in parallel)")
    print(f"{'='*80}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Create V3 optimizer
        opt = TritonZeroOrderOptimizerV3(
            model,
            n_perturbations=total_perturbations,
            pert_batch_size=pert_batch_size,
            learning_rate=1e-4,
            epsilon=1e-4,
        )

        # Run training steps
        print(f"\nRunning {num_steps} steps...")
        times = []

        for step in range(num_steps):
            batch = torch.randint(0, 100277, (data_batch_size, seq_len), device='cuda')

            t_start = time.time()
            result = opt.step(None, batch)
            torch.cuda.synchronize()
            t_end = time.time()

            step_time = t_end - t_start
            times.append(step_time)

            mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Step {step}: {step_time:.2f}s, {mem_gb:.2f}GB, loss={result['loss']:.4f}")

        # Calculate metrics
        avg_time = sum(times) / len(times)
        throughput = total_perturbations / avg_time
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"\n✓ SUCCESS!")
        print(f"  Avg time/step: {avg_time:.2f}s")
        print(f"  Throughput:    {throughput:.1f} perturbations/sec")
        print(f"  Peak memory:   {peak_mem_gb:.2f} GB")

        torch.cuda.empty_cache()

        return {
            'pert_batch_size': pert_batch_size,
            'success': True,
            'avg_time': avg_time,
            'throughput': throughput,
            'peak_mem_gb': peak_mem_gb,
        }

    except RuntimeError as e:
        print(f"\n✗ ERROR: {str(e)[:200]}")
        torch.cuda.empty_cache()
        return {
            'pert_batch_size': pert_batch_size,
            'success': False,
            'error': str(e),
        }


def main():
    print("="*80)
    print("V3 OPTIMIZER - SMALL MODEL TEST (Triton kernel debugging)")
    print("="*80)
    print("\nTesting vmap + functional_call + HybridGRU (Triton)\n")

    # Load smaller model for faster testing
    print("Loading small model (depth=6, dim=768)...")
    model = load_small_model(depth=6, dim=768)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Test configuration
    total_perturbations = 32

    # Test different parallel batch sizes
    batch_sizes_to_test = [1, 2, 4, 8, 16, 32]

    results = []

    for pert_batch_size in batch_sizes_to_test:
        result = test_v3_optimizer(
            model=model,
            pert_batch_size=pert_batch_size,
            total_perturbations=total_perturbations,
            num_steps=2,
            data_batch_size=2,
            seq_len=256,
        )
        results.append(result)

        if not result['success']:
            print(f"\n⚠ Hit error at pert_batch_size={pert_batch_size}")
            print(f"Error: {result.get('error', 'Unknown')[:200]}")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: V3 Small Model Test")
    print("="*80)

    print(f"\n{'Batch Size':>12} {'Status':>10} {'Time/step':>12} {'Throughput':>15} {'Memory':>10}")
    print("-" * 70)

    for r in results:
        if r['success']:
            status = "✓ SUCCESS"
            print(f"{r['pert_batch_size']:12d} {status:>10} {r['avg_time']:12.2f}s {r['throughput']:15.1f}/s {r['peak_mem_gb']:10.2f}GB")
        else:
            print(f"{r['pert_batch_size']:12d} {'✗ ERROR':>10} {'N/A':>12} {'N/A':>15} {'N/A':>10}")

    # Find optimal
    successful = [r for r in results if r['success']]
    if successful:
        optimal = max(successful, key=lambda x: x['throughput'])
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMAL: pert_batch_size = {optimal['pert_batch_size']}")
        print(f"   Throughput: {optimal['throughput']:.1f} perturbations/sec")
        print(f"   Time/step: {optimal['avg_time']:.2f}s")
        print(f"   Memory: {optimal['peak_mem_gb']:.2f} GB")
        print(f"{'='*80}")

        # Compare to serial
        if len(results) > 0 and results[0]['success']:
            serial = results[0]
            speedup = serial['avg_time'] / optimal['avg_time']
            print(f"\nSpeedup vs serial (batch=1): {speedup:.1f}×")
    else:
        print("\n⚠ All tests failed. This suggests an issue with vmap + functional_call + Triton")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
500M Model Virtual Perturbation Parallel Tuning

Tests the VirtualZeroOrderOptimizer with the real 500M model
to find optimal parallel batch size with seed-based perturbations.
"""

import torch
import torch.nn as nn
import sys
import time

# Import our virtual perturbation optimizer
from zero_order_virtual import VirtualZeroOrderOptimizer


def load_500m_model():
    """Load the 500M model architecture"""
    sys.path.insert(0, 'mingru')
    from minLM import minLM

    config = {
        'num_tokens': 100277,  # cl100k_base tokenizer
        'dim': 1536,
        'depth': 12,
        'expansion': 1.0,
        'conv_kernel_size': 4,
        'dropout': 0.0,
        'use_standard_gru': True,  # Use the working implementation
        'ff_mult': 0.0,
    }

    model = minLM(**config).cuda()
    model.eval()

    return model


def test_virtual_zero_order(
    model,
    pert_batch_size,
    total_perturbations=96,
    num_steps=3,
    data_batch_size=4,
    seq_len=512,
):
    """Test with specific perturbation batch size"""
    print(f"\n{'='*80}")
    print(f"TESTING: Virtual ZO with pert_batch_size = {pert_batch_size}")
    print(f"  (processing {pert_batch_size} perturbations at a time)")
    print(f"{'='*80}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Create virtual optimizer
        opt = VirtualZeroOrderOptimizer(
            model,
            learning_rate=1e-4,
            epsilon=1e-4,
            n_perturbations=total_perturbations,
            pert_batch_size=pert_batch_size,
            base_seed=42,
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
        if 'out of memory' in str(e):
            print(f"\n✗ OUT OF MEMORY!")
            torch.cuda.empty_cache()
            return {
                'pert_batch_size': pert_batch_size,
                'success': False,
                'avg_time': float('inf'),
                'throughput': 0,
                'peak_mem_gb': float('inf'),
            }
        raise


def main():
    print("="*80)
    print("500M MODEL - VIRTUAL PERTURBATION PARALLEL BATCH SIZE TUNING")
    print("="*80)
    print("\nSeed-based perturbations: O(1) memory per perturbation!")
    print("Testing optimal pert_batch_size for maximum throughput\n")

    # Load model
    print("Loading 500M model...")
    model = load_500m_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Test configuration
    total_perturbations = 96

    # Calculate theoretical memory savings
    old_memory_gb = (total_perturbations * num_params * 4) / (1024**3)
    new_memory_bytes = total_perturbations * 4
    print(f"\nMemory savings from virtual perturbations:")
    print(f"  Old (materialized): {old_memory_gb:.2f} GB")
    print(f"  New (seed-based):   {new_memory_bytes} bytes")
    print(f"  Reduction:          {old_memory_gb * 1024**3 / new_memory_bytes:.0f}× smaller!")

    # Test different parallel batch sizes
    # Start conservative since we're modifying params in-place
    batch_sizes_to_test = [1, 2, 4, 8, 16, 32, 48, 64, 96]

    results = []

    for pert_batch_size in batch_sizes_to_test:
        result = test_virtual_zero_order(
            model=model,
            pert_batch_size=pert_batch_size,
            total_perturbations=total_perturbations,
            num_steps=3,
            data_batch_size=4,
            seq_len=512,
        )
        results.append(result)

        if not result['success']:
            print(f"\n⚠ Hit limit at pert_batch_size={pert_batch_size}")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: 500M Model Virtual Perturbation Tuning")
    print("="*80)

    print(f"\n{'Batch Size':>12} {'Status':>10} {'Time/step':>12} {'Throughput':>15} {'Memory':>10}")
    print("-" * 70)

    for r in results:
        if r['success']:
            status = "✓ SUCCESS"
            print(f"{r['pert_batch_size']:12d} {status:>10} {r['avg_time']:12.2f}s {r['throughput']:15.1f}/s {r['peak_mem_gb']:10.2f}GB")
        else:
            print(f"{r['pert_batch_size']:12d} {'✗ LIMIT':>10} {'N/A':>12} {'N/A':>15} {'N/A':>10}")

    # Find optimal
    successful = [r for r in results if r['success']]
    if successful:
        optimal = max(successful, key=lambda x: x['throughput'])
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMAL FOR 500M MODEL: pert_batch_size = {optimal['pert_batch_size']}")
        print(f"   Throughput: {optimal['throughput']:.1f} perturbations/sec")
        print(f"   Time/step: {optimal['avg_time']:.2f}s")
        print(f"   Memory: {optimal['peak_mem_gb']:.2f} GB")
        print(f"{'='*80}")

        # Compare to serial
        if len(results) > 0 and results[0]['success']:
            serial = results[0]
            speedup = serial['avg_time'] / optimal['avg_time']
            print(f"\nSpeedup vs serial (batch=1): {speedup:.1f}×")


if __name__ == '__main__':
    main()

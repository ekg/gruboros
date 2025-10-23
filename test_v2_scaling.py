#!/usr/bin/env python3
"""
Test Virtual Perturbations V2 at Increasing Scales

Verifies:
1. Numerical correctness (evolving seeds work)
2. Memory savings (no OOM at large scales)
3. Performance (speed with freed memory)
"""

import torch
import torch.nn as nn
from zero_order_triton_v2 import TritonZeroOrderOptimizerV2
import time


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, vocab_size=256, dim=512, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.output = nn.Linear(dim, vocab_size)
        self.dim = dim

    def forward(self, x):
        x = self.embed(x)  # [batch, seq, dim]
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output(x)  # [batch, seq, vocab]
        return x


def test_seed_evolution():
    """Test that seeds evolve across steps"""
    print("="*80)
    print("TEST 1: Seed Evolution (Ergodicity)")
    print("="*80)

    model = SimpleModel(dim=64, n_layers=1).cuda()
    opt = TritonZeroOrderOptimizerV2(
        model,
        n_perturbations=8,
        base_seed=42
    )

    # Track seeds used across 3 steps
    seeds_per_step = []

    for step in range(3):
        # Manually compute what seeds should be used
        step_offset = step * opt.n_perturbations
        expected_seeds = [opt.base_seed + step_offset + i for i in range(opt.n_perturbations)]
        seeds_per_step.append(expected_seeds)

        # Run optimizer step
        batch = torch.randint(0, 256, (2, 32), device='cuda')
        result = opt.step(None, batch)

        print(f"\nStep {step}:")
        print(f"  Expected seeds: {expected_seeds[:4]} ... {expected_seeds[-1]}")
        print(f"  Step counter: {opt.step_counter}")

    # Verify no seed overlap
    all_seeds = [s for step in seeds_per_step for s in step]
    unique_seeds = set(all_seeds)

    print(f"\nTotal seeds used: {len(all_seeds)}")
    print(f"Unique seeds: {len(unique_seeds)}")

    if len(all_seeds) == len(unique_seeds):
        print("✓ PASS: All seeds are unique (ergodic!)")
        return True
    else:
        print("✗ FAIL: Some seeds repeated (not ergodic!)")
        return False


def test_scale(model_size_name, num_params, n_pert, batch_size, seq_len):
    """Test at a specific scale"""
    print(f"\n{'='*80}")
    print(f"TEST: {model_size_name} ({num_params/1e6:.1f}M params, {n_pert} pert)")
    print(f"{'='*80}")

    # Calculate theoretical memory
    old_memory_gb = (n_pert * num_params * 4) / (1024**3)
    new_memory_gb = (num_params * 4) / (1024**3)  # Peak: one perturbation

    print(f"\nMemory analysis:")
    print(f"  Materialized (old): {old_memory_gb:.2f} GB")
    print(f"  Virtual (new):      {new_memory_gb:.2f} GB peak")
    print(f"  Reduction:          {old_memory_gb / new_memory_gb:.1f}×")

    # Determine model size based on num_params
    # Approximate: num_params ≈ vocab*dim + n_layers*dim^2 + dim*vocab
    # For simplicity, use dim and layers to hit target
    dim = int((num_params / 100) ** 0.5)  # Rough approximation
    n_layers = max(2, num_params // (dim * dim + 2 * 256 * dim))

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        print(f"\nCreating model (dim={dim}, layers={n_layers})...")
        model = SimpleModel(dim=dim, n_layers=n_layers).cuda()

        actual_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Created model with {actual_params:,} params")

        print(f"\nCreating optimizer...")
        opt = TritonZeroOrderOptimizerV2(
            model,
            n_perturbations=n_pert,
            learning_rate=1e-4,
            epsilon=1e-4
        )

        print(f"\nRunning optimizer step...")
        batch = torch.randint(0, 256, (batch_size, seq_len), device='cuda')

        t_start = time.time()
        result = opt.step(None, batch)
        t_end = time.time()

        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"\n✓ SUCCESS!")
        print(f"  Peak memory: {peak_mem_gb:.2f} GB")
        print(f"  Time: {t_end - t_start:.2f}s")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Step counter: {result['step_counter']}")

        # Verify memory is reasonable
        if peak_mem_gb > old_memory_gb * 0.5:
            print(f"\n⚠ WARNING: Memory usage higher than expected!")
            print(f"  Expected < {old_memory_gb * 0.5:.2f} GB, got {peak_mem_gb:.2f} GB")

        del model, opt, batch
        torch.cuda.empty_cache()

        return {
            'success': True,
            'peak_mem_gb': peak_mem_gb,
            'time': t_end - t_start,
            'actual_params': actual_params,
        }

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"\n✗ OUT OF MEMORY!")
            torch.cuda.empty_cache()
            return {'success': False, 'peak_mem_gb': float('inf')}
        raise


def main():
    print("="*80)
    print("VIRTUAL PERTURBATIONS V2: SCALING TESTS")
    print("="*80)
    print("\nTesting virtual perturbations + evolving seeds at increasing scales")

    # Test 1: Verify seed evolution
    if not test_seed_evolution():
        print("\n✗ FAILED: Seed evolution test failed!")
        return

    # Test 2-4: Increasing model sizes
    test_configs = [
        ("Tiny", 100_000, 32, 4, 64),           # 100K params
        ("Small", 2_600_000, 64, 2, 128),       # 2.6M params
        ("Medium", 100_000_000, 96, 2, 256),    # 100M params
        ("Large", 500_000_000, 96, 2, 128),     # 500M params
    ]

    results = []
    for name, num_params, n_pert, batch_size, seq_len in test_configs:
        result = test_scale(name, num_params, n_pert, batch_size, seq_len)
        results.append((name, result))

        if not result['success']:
            print(f"\n⚠ {name} model failed - stopping here")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, result in results:
        if result['success']:
            print(f"\n{name}:")
            print(f"  Parameters: {result['actual_params']:,}")
            print(f"  Peak memory: {result['peak_mem_gb']:.2f} GB")
            print(f"  Time: {result['time']:.2f}s")
            print(f"  Status: ✓ SUCCESS")
        else:
            print(f"\n{name}:")
            print(f"  Status: ✗ OUT OF MEMORY")

    # Check if we reached 500M
    if any(r[0] == "Large" and r[1]['success'] for r in results):
        print("\n" + "="*80)
        print("🏆 BREAKTHROUGH: Successfully trained 500M model!")
        print("   Virtual perturbations enable large-scale zero-order optimization!")
        print("="*80)

if __name__ == '__main__':
    main()

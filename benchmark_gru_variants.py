#!/usr/bin/env python
"""
Benchmark GRU optimization variants at production scale.

Compares:
- Baseline: HybridGRU(H=2048) - 109K tok/s
- Projected: ProjectedGRU(H_rec=1024, D=2048) - target: 150-200K tok/s
- Low-rank: LowRankGRU(H=2048, R=384) - target: 140-180K tok/s
- Combined: ProjectedLowRankGRU(H_rec=1024, R=256) - target: 200-300K tok/s

Usage:
    python benchmark_gru_variants.py --all
    python benchmark_gru_variants.py --variant projected --h_rec 1024
    python benchmark_gru_variants.py --variant lowrank --rank 384
"""

import torch
import time
import argparse
from typing import Dict, Optional
import sys

# Import variants
try:
    from mingru.hybrid_fused_gru import HybridFusedGRU
    print("✓ HybridFusedGRU imported")
except ImportError as e:
    print(f"✗ Failed to import HybridFusedGRU: {e}")
    HybridFusedGRU = None

try:
    from mingru.projected_gru import ProjectedGRU
    print("✓ ProjectedGRU imported")
except ImportError as e:
    print(f"✗ Failed to import ProjectedGRU: {e}")
    ProjectedGRU = None

try:
    from mingru.lowrank_gru import LowRankGRU
    print("✓ LowRankGRU imported")
except ImportError as e:
    print(f"✗ Failed to import LowRankGRU: {e}")
    LowRankGRU = None

# Placeholder for combined variant
ProjectedLowRankGRU = None

# Production config
DEFAULT_CONFIG = {
    'B': 90,      # Batch size
    'T': 512,     # Sequence length
    'D': 2048,    # Model dimension
    'H': 2048,    # Hidden dimension (baseline)
    'H_rec': 1024,  # Recurrent dimension (projected)
    'R': 384,     # Rank (low-rank factorization)
}


def benchmark_forward_pass(
    model: torch.nn.Module,
    B: int,
    T: int,
    D: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Benchmark forward pass throughput."""

    model = model.to(device).to(dtype).eval()

    # Create dummy input
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            out = model(x)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            out = model(x)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    tokens = B * T
    throughput = tokens / avg_time

    # Memory usage
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB

    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_tok_s': throughput,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
        'tokens_per_batch': tokens,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_variant(
    variant_name: str,
    model_class: type,
    config: Dict,
    model_kwargs: Dict,
    device: str = 'cuda',
) -> Optional[Dict]:
    """Benchmark a single GRU variant."""

    if model_class is None:
        print(f"\n{'='*70}")
        print(f"Variant: {variant_name} - NOT IMPLEMENTED YET")
        print(f"{'='*70}")
        return None

    print(f"\n{'='*70}")
    print(f"Variant: {variant_name}")
    print(f"{'='*70}")

    try:
        # Create model
        model = model_class(**model_kwargs)
        num_params = count_parameters(model)

        print(f"Model: {model}")
        print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Config: B={config['B']}, T={config['T']}, D={config['D']}")

        # Benchmark
        results = benchmark_forward_pass(
            model=model,
            B=config['B'],
            T=config['T'],
            D=config['D'],
            device=device,
        )

        # Print results
        print(f"\nResults:")
        print(f"  Throughput:      {results['throughput_tok_s']:,.0f} tok/s")
        print(f"  Latency (avg):   {results['avg_time_ms']:.2f} ms")
        print(f"  Latency (min):   {results['min_time_ms']:.2f} ms")
        print(f"  Latency (max):   {results['max_time_ms']:.2f} ms")
        print(f"  Latency (std):   {results['std_time_ms']:.2f} ms")
        print(f"  Memory (alloc):  {results['memory_allocated_gb']:.2f} GB")
        print(f"  Memory (reserved): {results['memory_reserved_gb']:.2f} GB")

        results['variant'] = variant_name
        results['num_params'] = num_params

        return results

    except Exception as e:
        print(f"ERROR benchmarking {variant_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_all(config: Dict, device: str = 'cuda'):
    """Benchmark all available variants."""

    results = {}

    # Variant 0: Baseline (HybridGRU)
    if HybridFusedGRU is not None:
        res = benchmark_variant(
            variant_name='Baseline (HybridGRU)',
            model_class=HybridFusedGRU,
            config=config,
            model_kwargs={
                'dim': config['D'],
                'expansion_factor': 1.0,
            },
            device=device,
        )
        if res:
            results['baseline'] = res

    # Variant 1: Projected
    if ProjectedGRU is not None:
        res = benchmark_variant(
            variant_name=f"Projected (H_rec={config['H_rec']})",
            model_class=ProjectedGRU,
            config=config,
            model_kwargs={
                'dim': config['D'],
                'h_recurrent': config['H_rec'],
            },
            device=device,
        )
        if res:
            results['projected'] = res

    # Variant 2: Low-rank
    if LowRankGRU is not None:
        res = benchmark_variant(
            variant_name=f"Low-rank (R={config['R']})",
            model_class=LowRankGRU,
            config=config,
            model_kwargs={
                'dim': config['D'],
                'rank': config['R'],
            },
            device=device,
        )
        if res:
            results['lowrank'] = res

    # Variant 3: Combined
    if ProjectedLowRankGRU is not None:
        res = benchmark_variant(
            variant_name=f"Combined (H_rec={config['H_rec']}, R={config['R']})",
            model_class=ProjectedLowRankGRU,
            config=config,
            model_kwargs={
                'dim': config['D'],
                'h_recurrent': config['H_rec'],
                'rank': config['R'],
            },
            device=device,
        )
        if res:
            results['combined'] = res

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<30} {'Throughput':>15} {'Speedup':>10} {'Memory':>10}")
    print(f"{'-'*70}")

    baseline_throughput = results.get('baseline', {}).get('throughput_tok_s', 1.0)

    for key in ['baseline', 'projected', 'lowrank', 'combined']:
        if key in results:
            res = results[key]
            throughput = res['throughput_tok_s']
            speedup = throughput / baseline_throughput
            memory = res['memory_allocated_gb']
            variant = res['variant']

            print(f"{variant:<30} {throughput:>12,.0f} tok/s {speedup:>9.2f}× {memory:>9.2f} GB")

    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark GRU variants')
    parser.add_argument('--variant', type=str, choices=['baseline', 'projected', 'lowrank', 'combined', 'all'],
                        default='all', help='Which variant to benchmark')
    parser.add_argument('--all', action='store_true', help='Benchmark all variants')

    # Config overrides
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['B'])
    parser.add_argument('--seq_len', type=int, default=DEFAULT_CONFIG['T'])
    parser.add_argument('--dim', type=int, default=DEFAULT_CONFIG['D'])
    parser.add_argument('--h_rec', type=int, default=DEFAULT_CONFIG['H_rec'])
    parser.add_argument('--rank', type=int, default=DEFAULT_CONFIG['R'])

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Update config
    config = DEFAULT_CONFIG.copy()
    config['B'] = args.batch_size
    config['T'] = args.seq_len
    config['D'] = args.dim
    config['H_rec'] = args.h_rec
    config['R'] = args.rank

    print(f"{'='*70}")
    print("GRU VARIANT BENCHMARK")
    print(f"{'='*70}")
    print(f"Config: B={config['B']}, T={config['T']}, D={config['D']}")
    print(f"        H_rec={config['H_rec']} (projected), R={config['R']} (low-rank)")
    print(f"Device: {args.device}")
    print(f"{'='*70}")

    if args.all or args.variant == 'all':
        results = benchmark_all(config, device=args.device)
    elif args.variant == 'baseline':
        if HybridFusedGRU is None:
            print("ERROR: HybridFusedGRU not available")
            sys.exit(1)
        benchmark_variant(
            'Baseline (HybridGRU)',
            HybridFusedGRU,
            config,
            {'dim': config['D'], 'expansion_factor': 1.0},
            device=args.device,
        )
    elif args.variant == 'projected':
        if ProjectedGRU is None:
            print("ERROR: ProjectedGRU not implemented yet")
            print("Next step: Implement mingru/projected_gru.py")
            sys.exit(1)
        benchmark_variant(
            f"Projected (H_rec={config['H_rec']})",
            ProjectedGRU,
            config,
            {'dim': config['D'], 'h_recurrent': config['H_rec']},
            device=args.device,
        )
    elif args.variant == 'lowrank':
        if LowRankGRU is None:
            print("ERROR: LowRankGRU not implemented yet")
            print("Next step: Implement mingru/lowrank_gru.py")
            sys.exit(1)
        benchmark_variant(
            f"Low-rank (R={config['R']})",
            LowRankGRU,
            config,
            {'dim': config['D'], 'rank': config['R']},
            device=args.device,
        )
    elif args.variant == 'combined':
        if ProjectedLowRankGRU is None:
            print("ERROR: ProjectedLowRankGRU not implemented yet")
            print("Next step: Implement mingru/projected_lowrank_gru.py")
            sys.exit(1)
        benchmark_variant(
            f"Combined (H_rec={config['H_rec']}, R={config['R']})",
            ProjectedLowRankGRU,
            config,
            {'dim': config['D'], 'h_recurrent': config['H_rec'], 'rank': config['R']},
            device=args.device,
        )


if __name__ == '__main__':
    main()

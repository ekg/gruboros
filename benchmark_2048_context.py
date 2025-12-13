#!/usr/bin/env python
"""
Benchmark 2048 context training with gradient checkpointing.

Tests different inner_chunk_size values and batch sizes to find optimal tok/s
while monitoring GPU memory usage.
"""

import torch
import torch.nn.functional as F
import time
import sys
import gc

# Test configurations - push VRAM limits!
# RTX 6000 Ada has 47.4GB, we were only using ~16GB before
CONFIGS = [
    # (inner_chunk_size, batch_size)
    # Large inner_chunk = less checkpointing overhead
    # Start with largest (no checkpointing) and work down
    (2048, 4),   # No checkpointing at all
    (2048, 6),
    (2048, 8),
    (2048, 10),
    (2048, 12),
    (2048, 14),
    (2048, 16),
    (2048, 20),
    (2048, 24),
    (2048, 28),
    (2048, 32),  # Try to push batch size
    (1024, 8),
    (1024, 12),
    (1024, 16),
    (1024, 20),
    (1024, 24),
    (1024, 28),
    (1024, 32),
    (512, 12),
    (512, 16),
    (512, 20),
    (512, 24),
    (512, 28),
    (512, 32),
    (256, 16),
    (256, 20),
    (256, 24),
    (256, 28),
    (256, 32),
]

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024

def get_gpu_memory_reserved():
    """Get reserved GPU memory in MB."""
    return torch.cuda.memory_reserved() / 1024 / 1024

def run_benchmark(inner_chunk_size, batch_size, chunk_size=2048, warmup_steps=3, test_steps=10):
    """Run benchmark with given configuration."""
    from mingru.cudnn_gru_mult import CuDNNGRU_MultLM

    torch.cuda.empty_cache()
    gc.collect()

    try:
        # Create model with same config as 1B training
        model = CuDNNGRU_MultLM(
            num_tokens=50281,  # tiktoken p50k_base
            dim=2048,
            depth=27,  # 1B param config
            expansion_factor=1.0,
            ff_mult=0.0,
            use_checkpointing=True,
            inner_chunk_size=inner_chunk_size,
        ).cuda().bfloat16()

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Create optimizer (no scaler needed for bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.033)

        # Create dummy data
        input_ids = torch.randint(0, 50281, (batch_size, chunk_size), device='cuda')

        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Model computes loss internally, returns loss directly
                loss = model(input_ids)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # Record peak memory after warmup
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Benchmark
        start_time = time.time()
        total_tokens = 0

        for step in range(test_steps):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Model computes loss internally
                loss = model(input_ids)
            loss.backward()
            optimizer.step()
            total_tokens += batch_size * (chunk_size - 1)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        tokens_per_sec = total_tokens / elapsed

        # Cleanup
        del model, optimizer, input_ids
        torch.cuda.empty_cache()
        gc.collect()

        return {
            'success': True,
            'inner_chunk': inner_chunk_size,
            'batch_size': batch_size,
            'chunk_size': chunk_size,
            'tokens_per_sec': tokens_per_sec,
            'peak_memory_mb': peak_memory,
            'loss': loss.item(),
            'num_params': num_params,
        }

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'success': False,
                'inner_chunk': inner_chunk_size,
                'batch_size': batch_size,
                'chunk_size': chunk_size,
                'error': 'OOM',
            }
        else:
            raise


def main():
    print("=" * 70)
    print("Benchmark: 2048 Context with Gradient Checkpointing")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    results = []

    for inner_chunk, batch_size in CONFIGS:
        print(f"Testing: inner_chunk={inner_chunk}, batch_size={batch_size}...", end=" ", flush=True)

        result = run_benchmark(inner_chunk, batch_size)
        results.append(result)

        if result['success']:
            print(f"OK - {result['tokens_per_sec']:.0f} T/s, {result['peak_memory_mb']:.0f} MB, loss={result['loss']:.4f}")
        else:
            print(f"FAILED - {result['error']}")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'inner_chunk':>12} {'batch_size':>10} {'T/s':>10} {'Memory MB':>12} {'Status':>10}")
    print("-" * 70)

    successful = [r for r in results if r['success']]

    for r in results:
        if r['success']:
            print(f"{r['inner_chunk']:>12} {r['batch_size']:>10} {r['tokens_per_sec']:>10.0f} {r['peak_memory_mb']:>12.0f} {'OK':>10}")
        else:
            print(f"{r['inner_chunk']:>12} {r['batch_size']:>10} {'---':>10} {'---':>12} {'OOM':>10}")

    if successful:
        best = max(successful, key=lambda x: x['tokens_per_sec'])
        print()
        print(f"BEST CONFIG: inner_chunk={best['inner_chunk']}, batch_size={best['batch_size']}")
        print(f"  Throughput: {best['tokens_per_sec']:.0f} tok/s")
        print(f"  Peak Memory: {best['peak_memory_mb']:.0f} MB ({best['peak_memory_mb']/1024:.1f} GB)")

        # Also report highest memory utilization config that's still successful
        max_mem = max(successful, key=lambda x: x['batch_size'])
        if max_mem != best:
            print()
            print(f"MAX BATCH CONFIG: inner_chunk={max_mem['inner_chunk']}, batch_size={max_mem['batch_size']}")
            print(f"  Throughput: {max_mem['tokens_per_sec']:.0f} tok/s")
            print(f"  Peak Memory: {max_mem['peak_memory_mb']:.0f} MB ({max_mem['peak_memory_mb']/1024:.1f} GB)")


if __name__ == '__main__':
    main()

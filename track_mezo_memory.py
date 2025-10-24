#!/usr/bin/env python3
"""
Simple memory tracking for MeZO with increasing batch sizes.
No profiler, just direct torch.cuda.memory_* calls.
"""

import torch
import torch.distributed as dist
import os

# Setup distributed
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

from mingru.minLM import minLM
from mezo_optimizer import MeZOOptimizer

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def track_memory(batch_size):
    """Track memory for one MeZO step with given batch size."""

    torch.cuda.reset_peak_memory_stats(rank)
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"BATCH_SIZE={batch_size}")
        print('='*80)

    # Model
    model = minLM(
        num_tokens=100277,
        dim=1536,
        depth=12,
        ff_mult=0.0,
        expansion=1.0,
        conv_kernel_size=4,
        dropout=0.0,
        use_causal_conv_gru=True
    ).to(device).to(torch.bfloat16)

    mem_model = torch.cuda.memory_allocated(rank)
    dist.barrier()

    # Optimizer
    optimizer = MeZOOptimizer(
        model=model,
        learning_rate=0.0001,
        epsilon=0.0001,
        num_perturbations=4,
        base_seed=42,
        rank=rank,
        world_size=world_size,
        momentum=0.9
    )

    mem_optimizer = torch.cuda.memory_allocated(rank)
    dist.barrier()

    # Batch
    seq_len = 2048
    batch_data = torch.randint(0, 100277, (batch_size, seq_len), device=device)

    mem_batch = torch.cuda.memory_allocated(rank)
    dist.barrier()

    if rank == 0:
        print(f"Model:     {format_bytes(mem_model)}")
        print(f"Optimizer: {format_bytes(mem_optimizer - mem_model)}")
        print(f"Batch:     {format_bytes(mem_batch - mem_optimizer)}")
        print()

    # One MeZO step
    def batch_provider():
        return batch_data

    result = optimizer.step(
        loss_fn=None,
        batch_provider=batch_provider
    )

    mem_after = torch.cuda.memory_allocated(rank)
    mem_peak = torch.cuda.max_memory_allocated(rank)
    dist.barrier()

    if rank == 0:
        print(f"After step:  {format_bytes(mem_after)}")
        print(f"Peak:        {format_bytes(mem_peak)}")
        print(f"Peak spike:  {format_bytes(mem_peak - mem_batch)}")
        print(f"Loss:        {result['loss']:.4f}")

    # Cleanup
    del model
    del optimizer
    del batch_data
    torch.cuda.empty_cache()
    dist.barrier()

    return mem_peak

if __name__ == '__main__':
    if rank == 0:
        print("="*80)
        print("MEMORY TRACKING: MeZO with increasing batch sizes")
        print(f"GPUs: {world_size}")
        print("="*80)

    # Test batch sizes
    batch_sizes = [1, 2, 4, 8, 12]

    results = []
    for bs in batch_sizes:
        try:
            peak = track_memory(bs)
            results.append((bs, peak))
        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"\nOOM at batch_size={bs}")
                break
            else:
                raise

    if rank == 0:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Batch Size':<15} {'Peak Memory':<20} {'Per-batch':<15}")
        print("-"*80)
        for bs, peak in results:
            per_batch = peak / bs if bs > 0 else 0
            print(f"{bs:<15} {format_bytes(peak):<20} {format_bytes(per_batch):<15}")

    dist.destroy_process_group()

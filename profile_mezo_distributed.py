#!/usr/bin/env python3
"""
Profile MeZO training with PyTorch profiler to find memory leaks.
Run with batch_size=1 to isolate memory usage.
"""

import torch
import torch.distributed as dist
import os
import sys

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

def profile_training():
    """Profile one MeZO training step with memory tracking."""

    if rank == 0:
        print("="*80)
        print(f"PROFILING MeZO TRAINING (batch_size=1, {world_size} GPUs)")
        print("="*80)
        print()

    torch.cuda.reset_peak_memory_stats(rank)
    torch.cuda.empty_cache()

    # Create model
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

    after_model = torch.cuda.memory_allocated(rank)
    if rank == 0:
        print(f"1. Model created: {format_bytes(after_model)}")

    # Create optimizer
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

    after_optimizer = torch.cuda.memory_allocated(rank)
    if rank == 0:
        print(f"2. Optimizer created: {format_bytes(after_optimizer - after_model)}")
        print()

    # Create tiny batch
    batch_size = 1
    seq_len = 2048
    batch_data = torch.randint(0, 100277, (batch_size, seq_len), device=device)

    after_batch = torch.cuda.memory_allocated(rank)
    if rank == 0:
        print(f"3. Batch created (size={batch_size}): {format_bytes(after_batch - after_optimizer)}")
        print()

    # Profile one MeZO step with memory tracking
    if rank == 0:
        print("4. Running ONE MeZO step with profiler...")
        print()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # One MeZO step: K=4 perturbations × 2 directions = 8 forward passes
        def batch_provider():
            return batch_data

        result = optimizer.step(
            loss_fn=None,
            batch_provider=batch_provider
        )

    after_step = torch.cuda.memory_allocated(rank)
    peak_step = torch.cuda.max_memory_allocated(rank)

    if rank == 0:
        print(f"5. After MeZO step:")
        print(f"   Current: {format_bytes(after_step)}")
        print(f"   Peak: {format_bytes(peak_step)}")
        print(f"   Step added: {format_bytes(after_step - after_batch)}")
        print(f"   Peak spike: {format_bytes(peak_step - after_batch)}")
        print()

        # Print memory-heavy operations
        print("="*80)
        print("TOP MEMORY ALLOCATIONS")
        print("="*80)
        print()

        # Sort by memory
        events = prof.key_averages()
        memory_events = [(e.self_cpu_memory_usage + e.self_cuda_memory_usage, e) for e in events]
        memory_events.sort(reverse=True)

        print(f"{'Operation':<50} {'Memory':<15} {'Count':<8}")
        print("-"*80)
        for mem, event in memory_events[:30]:
            if mem > 0:
                op_name = event.key[:47] + "..." if len(event.key) > 50 else event.key
                print(f"{op_name:<50} {format_bytes(mem):<15} {event.count:<8}")

        print()
        print("="*80)
        print("SAVING TRACE TO: /tmp/mezo_profile_trace.json")
        print("View with: chrome://tracing")
        print("="*80)

        # Export trace
        prof.export_chrome_trace("/tmp/mezo_profile_trace.json")

        # Also print memory timeline
        print()
        print("="*80)
        print("MEMORY TIMELINE")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=30
        ))

    dist.barrier()

if __name__ == '__main__':
    profile_training()
    dist.destroy_process_group()

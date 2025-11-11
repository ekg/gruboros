#!/usr/bin/env python
"""
Find maximum batch size for ProjectedGRU at production config.

Config:
- T=512 (chunk size)
- D=2048 (model dim)
- H_rec=1280 (recurrent dim)
- depth=20 layers
"""

import torch
import sys
from mingru.projected_gru import ProjectedGRU

def test_batch_size(batch_size, seq_len=512, dim=2048, h_rec=1280, device='cuda'):
    """Test if batch_size fits in memory."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        model = ProjectedGRU(
            dim=dim,
            h_recurrent=h_rec,
            expansion_factor=1.0,
        ).to(device).to(torch.bfloat16)

        # Create dummy input (requires_grad for backward test)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16, requires_grad=True)

        # Forward pass
        output = model(x)

        # Backward pass (training simulation)
        loss = output.sum()
        loss.backward()

        # Check memory
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        usage_pct = (peak_memory / total_memory) * 100

        print(f"B={batch_size:3d}: Peak={peak_memory:.2f}GB ({usage_pct:.1f}% of {total_memory:.0f}GB) ✓")

        # Clean up
        del model, x, output, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return True, peak_memory

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"B={batch_size:3d}: OOM ✗")
            torch.cuda.empty_cache()
            return False, None
        else:
            raise

def binary_search_max_batch(min_b=1, max_b=200, **kwargs):
    """Binary search to find maximum batch size."""

    print(f"\n{'='*70}")
    print("Finding maximum batch size for ProjectedGRU")
    print(f"Config: T={kwargs['seq_len']}, D={kwargs['dim']}, H_rec={kwargs['h_rec']}")
    print(f"{'='*70}\n")

    # First check if min_b works
    works, mem = test_batch_size(min_b, **kwargs)
    if not works:
        print(f"\nERROR: Even B={min_b} doesn't fit!")
        return None

    # Binary search
    last_working = min_b
    last_memory = mem

    while min_b <= max_b:
        mid = (min_b + max_b) // 2

        works, mem = test_batch_size(mid, **kwargs)

        if works:
            last_working = mid
            last_memory = mem
            min_b = mid + 1  # Try larger
        else:
            max_b = mid - 1  # Try smaller

    print(f"\n{'='*70}")
    print(f"Maximum batch size: {last_working}")
    print(f"Peak memory usage: {last_memory:.2f} GB")
    print(f"{'='*70}\n")

    return last_working

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = 'cuda'
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total memory: {total_memory:.1f} GB")

    # Production config
    config = {
        'seq_len': 512,
        'dim': 2048,
        'h_rec': 1280,
        'device': device,
    }

    # Find max batch size
    max_batch = binary_search_max_batch(min_b=90, max_b=150, **config)

    if max_batch:
        print(f"\nRecommendation:")
        print(f"  Max safe batch: {max_batch}")
        print(f"  Recommended: {int(max_batch * 0.95)} (95% of max for safety)")
        print(f"\n  For training with grad_accum=16:")
        print(f"    Effective batch: {int(max_batch * 0.95) * 16 * 8} tokens/update (8 GPUs)")
        print(f"    That's {int(max_batch * 0.95) * 16 * 8 * 512 / 1e6:.1f}M tokens/update")

if __name__ == '__main__':
    main()

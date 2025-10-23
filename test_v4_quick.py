#!/usr/bin/env python3
"""
Quick test of V4 optimizer to verify Triton kernel compatibility
"""

import torch
import sys
import time

from zero_order_triton_v4 import TritonZeroOrderOptimizerV4

def load_tiny_model():
    """Load tiny model for quick testing"""
    sys.path.insert(0, 'mingru')
    from minLM import minLM

    config = {
        'num_tokens': 100277,
        'dim': 384,  # Tiny
        'depth': 2,  # Very shallow
        'expansion': 1.0,
        'conv_kernel_size': 4,
        'dropout': 0.0,
        'use_hybrid_gru': True,  # Use Triton kernel - this is the test!
        'ff_mult': 0.0,
    }

    model = minLM(**config).cuda()
    model.eval()
    return model


def main():
    print("="*80)
    print("V4 OPTIMIZER - TRITON KERNEL COMPATIBILITY TEST")
    print("="*80)
    print("\nTesting if V4 works with HybridGRU (Triton kernel)...\n")

    # Load tiny model
    print("Loading tiny model (depth=2, dim=384)...")
    model = load_tiny_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Create V4 optimizer
    print("\nCreating V4 optimizer...")
    opt = TritonZeroOrderOptimizerV4(
        model,
        n_perturbations=8,  # Just a few for quick test
        pert_batch_size=4,
        learning_rate=1e-4,
        epsilon=1e-4,
    )

    # Run one step
    print("\nRunning one training step...")
    batch = torch.randint(0, 100277, (2, 128), device='cuda')

    try:
        t_start = time.time()
        result = opt.step(None, batch)
        torch.cuda.synchronize()
        t_end = time.time()

        print(f"\n✓ SUCCESS! V4 works with Triton kernels!")
        print(f"  Time: {t_end - t_start:.2f}s")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"\nThis confirms V4 is compatible with HybridFusedGRU (Triton)")

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)[:200]}")
        print(f"\nV4 does NOT work with Triton kernels")
        raise


if __name__ == '__main__':
    main()

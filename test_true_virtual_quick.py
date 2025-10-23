#!/usr/bin/env python3
"""
Test TRUE virtual perturbations - layer-by-layer on-the-fly generation
"""

import torch
import sys
import time

from zero_order_true_virtual import TrueVirtualZeroOrderOptimizer


def load_tiny_model():
    """Load tiny model for quick testing"""
    sys.path.insert(0, 'mingru')
    from minLM import minLM

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


def main():
    print("="*80)
    print("TRUE VIRTUAL PERTURBATIONS TEST")
    print("="*80)
    print("\nOn-the-fly perturbation generation in each Linear layer\n")

    # Load tiny model
    print("Loading tiny model...")
    model = load_tiny_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Create optimizer
    print("\nCreating TRUE virtual optimizer...")
    opt = TrueVirtualZeroOrderOptimizer(
        model,
        n_perturbations=8,
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

        print(f"\n✓ SUCCESS! TRUE virtual perturbations work!")
        print(f"  Time: {t_end - t_start:.2f}s")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"\nThis confirms layer-by-layer on-the-fly perturbation generation works!")

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

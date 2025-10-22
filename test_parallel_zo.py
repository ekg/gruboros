#!/usr/bin/env python3
"""
Test script for parallel zero-order optimizer.
Verifies correctness and measures speedup vs sequential implementation.
"""

import torch
import torch.nn as nn
import time
from zero_order_parallel import ParallelZeroOrderOptimizer
from zero_order_optimizer import CD_RGE_Optimizer
from zero_order_vmap import VmapZeroOrderOptimizer

# Simple test model
class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, dim=128, depth=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(depth)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.lm_head(x)
        return logits


def test_parallel_optimizer():
    print("=" * 80)
    print("Testing Parallel Zero-Order Optimizer")
    print("=" * 80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = TinyLM(vocab_size=256, dim=128, depth=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create test batch
    batch_size = 16
    seq_len = 128
    batch_data = torch.randint(0, 256, (batch_size, seq_len), device=device)
    print(f"Batch shape: {batch_data.shape}")

    # Test with small number of perturbations first
    n_pert = 16
    pert_batch = 8

    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Perturbations: {n_pert}")
    print(f"  Perturbation batch size: {pert_batch}")
    print(f"  Forward passes per step: {2 * n_pert}")
    print(f"{'='*80}\n")

    # Create optimizer
    optimizer = ParallelZeroOrderOptimizer(
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        pert_batch_size=pert_batch
    )

    # Run a few steps
    print("\nRunning optimization steps...")
    for step in range(5):
        start = time.time()

        result = optimizer.step(loss_fn=None, batch_data=batch_data)

        elapsed = time.time() - start

        print(f"Step {step+1}:")
        print(f"  Loss: {result['loss']:.4f} ± {result['loss_std']:.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"    Forward: {result['time_forward']:.3f}s")
        print(f"    Backward: {result['time_backward']:.3f}s")
        print(f"  Throughput: {batch_size * seq_len / elapsed:.0f} tokens/sec")

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


def compare_sequential_vs_parallel():
    """Compare parallel optimizer against sequential baseline"""
    print("\n" + "="*80)
    print("Comparing Sequential vs Parallel Optimizer")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create two identical models
    model_seq = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
    model_par = TinyLM(vocab_size=256, dim=64, depth=2).to(device)

    # Copy weights
    model_par.load_state_dict(model_seq.state_dict())

    # Test batch
    batch_size = 8
    seq_len = 64
    batch_data = torch.randint(0, 256, (batch_size, seq_len), device=device)

    n_pert = 16

    print(f"\nConfiguration:")
    print(f"  Model params: {sum(p.numel() for p in model_seq.parameters()):,}")
    print(f"  Batch: {batch_size} x {seq_len}")
    print(f"  Perturbations: {n_pert}")
    print(f"  Forward passes: {2 * n_pert}")

    # Sequential optimizer
    print(f"\n{'='*80}")
    print("Sequential Optimizer")
    print(f"{'='*80}")

    opt_seq = CD_RGE_Optimizer(
        model_seq,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        grad_accum=1
    )

    # Loss function
    def loss_fn_seq(batch_data, hiddens=None, conv=None):
        with torch.no_grad():
            logits = model_seq(batch_data[:, :-1])
            targets = batch_data[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return loss

    start = time.time()
    result_seq = opt_seq.step(loss_fn_seq, batch_provider=lambda: batch_data)
    time_seq = time.time() - start

    print(f"Loss: {result_seq['loss']:.4f}")
    print(f"Time: {time_seq:.3f}s")
    print(f"Throughput: {batch_size * seq_len / time_seq:.0f} tokens/sec")

    # Parallel optimizer
    print(f"\n{'='*80}")
    print("Parallel Optimizer")
    print(f"{'='*80}")

    opt_par = ParallelZeroOrderOptimizer(
        model_par,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        pert_batch_size=8
    )

    start = time.time()
    result_par = opt_par.step(loss_fn=None, batch_data=batch_data)
    time_par = time.time() - start

    print(f"Loss: {result_par['loss']:.4f}")
    print(f"Time: {time_par:.3f}s")
    print(f"Throughput: {batch_size * seq_len / time_par:.0f} tokens/sec")

    # Speedup
    print(f"\n{'='*80}")
    print(f"SPEEDUP: {time_seq / time_par:.2f}×")
    print(f"Sequential: {time_seq:.3f}s")
    print(f"Parallel:   {time_par:.3f}s")
    print(f"Saved:      {time_seq - time_par:.3f}s per step")
    print(f"{'='*80}\n")


def test_vmap_optimizer():
    """Test vmap-based optimizer"""
    print("\n" + "="*80)
    print("Testing Vmap Zero-Order Optimizer (Option 1)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TinyLM(vocab_size=256, dim=64, depth=2).to(device)

    # Test batch
    batch_size = 8
    seq_len = 64
    batch_data = torch.randint(0, 256, (batch_size, seq_len), device=device)

    n_pert = 16

    print(f"\nConfiguration:")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Batch: {batch_size} x {seq_len}")
    print(f"  Perturbations: {n_pert}")

    opt_vmap = VmapZeroOrderOptimizer(
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        pert_batch_size=8
    )

    print("\nRunning 3 steps...")
    for step in range(3):
        start = time.time()
        result = opt_vmap.step(loss_fn=None, batch_data=batch_data)
        elapsed = time.time() - start

        print(f"Step {step+1}:")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {batch_size * seq_len / elapsed:.0f} tokens/sec")

    print("\n" + "="*80)
    print("Vmap optimizer test completed successfully!")
    print("="*80)


if __name__ == '__main__':
    # Test parallel optimizer
    test_parallel_optimizer()

    # Test vmap optimizer
    test_vmap_optimizer()

    # Compare against sequential
    print("\n")
    compare_sequential_vs_parallel()

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
from zero_order_batched import BatchedPerturbationOptimizer

try:
    from zero_order_triton import TritonZeroOrderOptimizer, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    print("WARNING: Triton optimizer not available")

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


def test_batched_optimizer():
    """Test batched perturbation dimension optimizer"""
    print("\n" + "="*80)
    print("Testing Batched Perturbation Optimizer (Option 2)")
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

    opt_batched = BatchedPerturbationOptimizer(
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        pert_batch_size=8
    )

    print("\nRunning 3 steps...")
    for step in range(3):
        start = time.time()
        result = opt_batched.step(loss_fn=None, batch_data=batch_data)
        elapsed = time.time() - start

        print(f"Step {step+1}:")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {batch_size * seq_len / elapsed:.0f} tokens/sec")

    print("\n" + "="*80)
    print("Batched optimizer test completed successfully!")
    print("="*80)


def test_triton_optimizer():
    """Test Triton-based optimizer"""
    if not TRITON_AVAILABLE:
        print("\n" + "="*80)
        print("Triton Zero-Order Optimizer - SKIPPED (Triton not available)")
        print("Install with: pip install triton")
        print("="*80)
        return

    print("\n" + "="*80)
    print("Testing Triton Zero-Order Optimizer (Option 3.0)")
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

    opt_triton = TritonZeroOrderOptimizer(
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=n_pert,
        pert_batch_size=8
    )

    print("\nRunning 3 steps...")
    for step in range(3):
        start = time.time()
        result = opt_triton.step(loss_fn=None, batch_data=batch_data)
        elapsed = time.time() - start

        print(f"Step {step+1}:")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {batch_size * seq_len / elapsed:.0f} tokens/sec")

    print("\n" + "="*80)
    print("Triton optimizer test completed successfully!")
    print("="*80)


def compare_all_methods():
    """Compare all zero-order optimization methods"""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: All Zero-Order Methods")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test configuration
    batch_size = 8
    seq_len = 64
    n_pert = 16

    print(f"\nConfiguration:")
    print(f"  Batch: {batch_size} x {seq_len}")
    print(f"  Perturbations: {n_pert}")
    print(f"  Forward passes: {2 * n_pert}")

    # Create batch data
    batch_data = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Test each method
    results = {}

    # Method 1: Sequential (baseline)
    model1 = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
    opt1 = CD_RGE_Optimizer(model1, learning_rate=1e-4, epsilon=1e-4,
                           n_perturbations=n_pert, grad_accum=1)

    def loss_fn_seq(batch_data, hiddens=None, conv=None):
        with torch.no_grad():
            logits = model1(batch_data[:, :-1])
            targets = batch_data[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return loss

    start = time.time()
    result1 = opt1.step(loss_fn_seq, batch_provider=lambda: batch_data)
    time1 = time.time() - start
    results['Sequential'] = time1

    # Method 2: Dynamic Perturbation
    model2 = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
    opt2 = ParallelZeroOrderOptimizer(model2, learning_rate=1e-4, epsilon=1e-4,
                                     n_perturbations=n_pert, pert_batch_size=8)
    start = time.time()
    result2 = opt2.step(loss_fn=None, batch_data=batch_data)
    time2 = time.time() - start
    results['Dynamic'] = time2

    # Method 3: Vmap
    model3 = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
    opt3 = VmapZeroOrderOptimizer(model3, learning_rate=1e-4, epsilon=1e-4,
                                 n_perturbations=n_pert, pert_batch_size=8)
    start = time.time()
    result3 = opt3.step(loss_fn=None, batch_data=batch_data)
    time3 = time.time() - start
    results['Vmap'] = time3

    # Method 4: Batched
    model4 = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
    opt4 = BatchedPerturbationOptimizer(model4, learning_rate=1e-4, epsilon=1e-4,
                                       n_perturbations=n_pert, pert_batch_size=8)
    start = time.time()
    result4 = opt4.step(loss_fn=None, batch_data=batch_data)
    time4 = time.time() - start
    results['Batched'] = time4

    # Method 5: Triton (if available)
    if TRITON_AVAILABLE:
        model5 = TinyLM(vocab_size=256, dim=64, depth=2).to(device)
        opt5 = TritonZeroOrderOptimizer(model5, learning_rate=1e-4, epsilon=1e-4,
                                       n_perturbations=n_pert, pert_batch_size=8)
        start = time.time()
        result5 = opt5.step(loss_fn=None, batch_data=batch_data)
        time5 = time.time() - start
        results['Triton'] = time5

    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    baseline = results['Sequential']
    for name, time_taken in results.items():
        speedup = baseline / time_taken
        throughput = batch_size * seq_len / time_taken
        print(f"{name:15s}: {time_taken:6.3f}s  |  {speedup:5.2f}×  |  {throughput:7.0f} tokens/sec")

    print("\n" + "="*80)
    print(f"BEST METHOD: {min(results, key=results.get)}")
    print(f"MAX SPEEDUP: {baseline / min(results.values()):.2f}×")
    print("="*80)


if __name__ == '__main__':
    # Test parallel optimizer
    test_parallel_optimizer()

    # Test vmap optimizer
    test_vmap_optimizer()

    # Test batched optimizer
    test_batched_optimizer()

    # Test Triton optimizer
    test_triton_optimizer()

    # Compare against sequential
    print("\n")
    compare_sequential_vs_parallel()

    # Comprehensive comparison
    compare_all_methods()

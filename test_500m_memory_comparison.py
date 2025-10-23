#!/usr/bin/env python3
"""
Test REAL 500M model: Memory comparison between implementations

Demonstrates:
1. Current Vmap/Triton implementations store perturbation VECTORS
2. This causes massive memory usage: O(P × N) where P=perturbations, N=parameters
3. Virtual (seed-based) perturbations use O(1) memory per perturbation

For 500M params, 96 perturbations:
- Materialized: 96 × 500M × 4 bytes = 192 GB
- Virtual (seeds): 96 × 4 bytes = 384 bytes
- Reduction: 500,000× smaller!
"""

import torch
import sys
from pathlib import Path

# Load checkpoint to get model config
checkpoint_path = "/mnt/nvme2n1/erikg/minlms/20250913_141622_500m_8dbbd28/archive_001.pt"

print("="*80)
print("LOADING 500M MODEL")
print("="*80)
print(f"\nCheckpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_config = checkpoint['model_config']
model_state = checkpoint['model_state_dict']

total_params = sum(p.numel() for p in model_state.values())
print(f"✓ Loaded model with {total_params:,} parameters ({total_params/1e6:.1f}M)")
print(f"\nModel config:")
for k, v in model_config.items():
    print(f"  {k}: {v}")

# Import model and optimizers
from mingru.minLM import minLM
from zero_order_vmap import VmapZeroOrderOptimizer
from zero_order_triton import TritonZeroOrderOptimizer

# Test configuration
n_pert = 96
batch_size = 2
seq_len = 128

print("\n" + "="*80)
print(f"MEMORY ANALYSIS: {n_pert} Perturbations")
print("="*80)

# Calculate theoretical memory for perturbation vectors
perturbation_memory_gb = (n_pert * total_params * 4) / (1024**3)
print(f"\nTheoretical memory for perturbation VECTORS:")
print(f"  {n_pert} × {total_params:,} × 4 bytes = {perturbation_memory_gb:.2f} GB")

# Seed-based memory
seed_memory_bytes = n_pert * 4
print(f"\nTheoretical memory for SEEDS:")
print(f"  {n_pert} × 4 bytes = {seed_memory_bytes} bytes ({seed_memory_bytes/1024:.2f} KB)")

print(f"\n🏆 Memory reduction: {perturbation_memory_gb*1024**3 / seed_memory_bytes:,.0f}× smaller!")

print("\n" + "="*80)
print("TESTING CURRENT IMPLEMENTATIONS")
print("="*80)
print("\nNOTE: Both Vmap and Triton currently store perturbation VECTORS")
print("      This is why they can't scale to 500M models!")

def test_implementation(optimizer_class, name):
    """Test a single implementation"""
    print(f"\n{'-'*80}")
    print(f"Testing {name}")
    print(f"{'-'*80}")

    device = torch.device('cuda')

    # Create model with correct config
    model = minLM(**model_config).to(device)

    # Load weights
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Loaded pretrained weights")
    except Exception as e:
        print(f"⚠ Could not load weights (using initialized model): {e}")

    try:
        # Create optimizer
        print(f"\nCreating {name} optimizer...")
        opt = optimizer_class(
            model,
            learning_rate=1e-4,
            epsilon=1e-4,
            n_perturbations=n_pert,
            pert_batch_size=min(16, n_pert)
        )

        print(f"Creating batch data ({batch_size} × {seq_len})...")
        batch_data = torch.randint(0, 256, (batch_size, seq_len), device=device)

        print("\nResetting memory stats...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        print("Running optimizer step...")
        result = opt.step(loss_fn=None, batch_data=batch_data)

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n✓ SUCCESS!")
        print(f"  Peak memory: {peak_mem_gb:.2f} GB")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Time: {result['time_total']:.2f}s")

        # Cleanup
        del model, opt, batch_data
        torch.cuda.empty_cache()

        return {
            'success': True,
            'peak_mem_gb': peak_mem_gb,
            'loss': result['loss'],
            'time': result['time_total']
        }

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"\n❌ OUT OF MEMORY!")
            print(f"   The implementation stores {perturbation_memory_gb:.2f} GB of perturbation vectors")
            print(f"   This exceeds GPU memory capacity")

            # Cleanup
            del model
            if 'opt' in locals():
                del opt
            if 'batch_data' in locals():
                del batch_data
            torch.cuda.empty_cache()

            return {
                'success': False,
                'peak_mem_gb': float('inf'),
                'error': 'OOM'
            }
        else:
            raise

# Test both implementations
print("\n⚠ WARNING: These tests may cause OOM on GPUs with < 80GB memory")
print("   because they store full perturbation vectors!")

vmap_result = test_implementation(VmapZeroOrderOptimizer, "Vmap (stores vectors)")
triton_result = test_implementation(TritonZeroOrderOptimizer, "Triton (stores vectors)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nPerturbation storage requirements:")
print(f"  Materialized (current): {perturbation_memory_gb:.2f} GB")
print(f"  Virtual (seeds only):   {seed_memory_bytes} bytes")
print(f"  Reduction:              {perturbation_memory_gb*1024**3 / seed_memory_bytes:,.0f}×")

print(f"\nResults:")
for name, result in [("Vmap", vmap_result), ("Triton", triton_result)]:
    if result['success']:
        print(f"  {name:8s}: ✓ Peak {result['peak_mem_gb']:.2f} GB, {result['time']:.2f}s")
    else:
        print(f"  {name:8s}: ✗ {result['error']}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nTo train 500M models with zero-order optimization, we MUST use")
print("virtual (seed-based) perturbations instead of storing vectors!")
print("\nThis reduces memory from ~192 GB to ~384 bytes for 96 perturbations.")
print("="*80)

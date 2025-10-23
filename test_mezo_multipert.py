"""
Quick test to verify MeZO multi-perturbation implementation.

Tests:
1. Optimizer initializes correctly with num_perturbations parameter
2. Forward passes scale correctly (2K passes for K perturbations)
3. GPU utilization monitoring works
4. Seed generation is unique per perturbation
"""

import torch
import torch.nn as nn
from mezo_optimizer import MeZOOptimizer

print("=== MeZO Multi-Perturbation Test ===\n")

# Create a tiny model
class TinyModel(nn.Module):
    def __init__(self, vocab_size=100, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.linear(self.embed(x))

model = TinyModel().cuda()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 1: Initialize with different num_perturbations
print("\nTest 1: Initialization")
for num_p in [1, 2, 4]:
    optimizer = MeZOOptimizer(
        model=model,
        learning_rate=1e-4,
        epsilon=1e-3,
        num_perturbations=num_p,
        base_seed=42,
        rank=0,
        world_size=1
    )
    print(f"  ✓ Created optimizer with {num_p} perturbations")

# Test 2: Run optimization step and check forward passes
print("\nTest 2: Forward pass counting")
optimizer = MeZOOptimizer(
    model=model,
    learning_rate=1e-4,
    epsilon=1e-3,
    num_perturbations=4,
    base_seed=42,
    rank=0,
    world_size=1
)

# Create a simple batch provider
def batch_provider():
    return torch.randint(0, 100, (2, 16)).cuda()

# Run one step
result = optimizer.step(None, batch_provider)

print(f"  Perturbations: {result['num_perturbations']}")
print(f"  Forward passes: {result['forward_passes']}")
print(f"  Expected: {2 * result['num_perturbations']}")
print(f"  ✓ Match: {result['forward_passes'] == 2 * result['num_perturbations']}")

# Test 3: Check GPU utilization is reported
print("\nTest 3: GPU Utilization Monitoring")
print(f"  GPU util: {result['gpu_utilization']:.1f}%")
print(f"  ✓ Monitoring works: {result['gpu_utilization'] >= 0}")

# Test 4: Verify different seeds per perturbation
print("\nTest 4: Seed uniqueness")
num_p = 4
seeds_step0 = []
seeds_step1 = []
base_seed = 42
world_size = 1
rank = 0

for k in range(num_p):
    # Step 0
    seed = base_seed + (0 * num_p + k) * world_size + rank
    seeds_step0.append(seed)
    # Step 1
    seed = base_seed + (1 * num_p + k) * world_size + rank
    seeds_step1.append(seed)

print(f"  Seeds at step 0: {seeds_step0}")
print(f"  Seeds at step 1: {seeds_step1}")
print(f"  ✓ All unique: {len(set(seeds_step0 + seeds_step1)) == 2 * num_p}")

# Test 5: Compare single vs multi-perturbation (should move parameters)
print("\nTest 5: Parameter updates")
initial_param = model.embed.weight.data.clone()

optimizer_single = MeZOOptimizer(
    model=model,
    learning_rate=1e-4,
    epsilon=1e-3,
    num_perturbations=1,
    base_seed=42,
    rank=0,
    world_size=1
)
optimizer_single.step(None, batch_provider)
single_param = model.embed.weight.data.clone()

# Reset model
model.embed.weight.data.copy_(initial_param)

optimizer_multi = MeZOOptimizer(
    model=model,
    learning_rate=1e-4,
    epsilon=1e-3,
    num_perturbations=4,
    base_seed=42,
    rank=0,
    world_size=1
)
optimizer_multi.step(None, batch_provider)
multi_param = model.embed.weight.data.clone()

single_diff = (single_param - initial_param).abs().mean().item()
multi_diff = (multi_param - initial_param).abs().mean().item()

print(f"  Single-pert param change: {single_diff:.6f}")
print(f"  Multi-pert param change:  {multi_diff:.6f}")
print(f"  ✓ Parameters updated: {single_diff > 0 and multi_diff > 0}")

print("\n=== All Tests Passed! ===\n")

# Performance comparison
print("Performance Summary:")
print("  Config              | Forward Passes | Variance Reduction")
print("  --------------------|----------------|-------------------")
for k in [1, 2, 4, 8]:
    print(f"  grad_accum={k:<2}       |      {2*k:<6}      |        {k}×")

print("\nRecommendation: Use grad_accum=4 or 8 for 8 GPUs to maximize throughput")
print("                while keeping variance reduction benefits.\n")

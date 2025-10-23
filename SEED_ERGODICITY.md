# Seed Ergodicity: Evolving Perturbations Across Steps

## The Problem

**Current Code (WRONG):**
```python
# In __init__
self.rank = rank
self.n_perturbations = 96

# In step()
start_idx = self.rank * pert_per_worker  # e.g., rank 0 → idx 0
perturbations = self.generate_perturbations(pert_per_worker, seed_offset=start_idx)
# → Uses seeds [0, 1, 2, ..., 95] EVERY SINGLE STEP!
```

**Issue:** Samples the SAME perturbation directions every step!
- Step 0: seeds [0, 1, ..., 95]
- Step 1: seeds [0, 1, ..., 95] ← SAME!
- Step 2: seeds [0, 1, ..., 95] ← SAME!
- Step 1000: seeds [0, 1, ..., 95] ← STILL SAME!

This violates **ergodicity** - we're not exploring the full parameter space!

## The Solution: Evolving Seeds

**Correct Code:**
```python
class VirtualZeroOrderOptimizer:
    def __init__(self, ...):
        self.base_seed = 42  # Initial RNG seed
        self.step_counter = 0  # Tracks optimization steps
        self.n_perturbations = 96
        self.rank = rank

    def step(self, ...):
        # Deterministic but EVOLVING seed sequence
        step_seed_offset = self.step_counter * self.n_perturbations
        start_idx = self.rank * pert_per_worker

        for i in range(pert_per_worker):
            # Seed evolves with step AND distributes across workers
            seed = self.base_seed + step_seed_offset + start_idx + i
            pert = self._generate_single_perturbation(seed)
            # ... use perturbation ...

        self.step_counter += 1  # Advance to next seed block
```

**Seed Timeline (single GPU, 96 perturbations):**
- Step 0: seeds [42, 43, ..., 137]
- Step 1: seeds [138, 139, ..., 233]
- Step 2: seeds [234, 235, ..., 329]
- Step t: seeds [42 + t×96, 42 + t×96 + 1, ..., 42 + (t+1)×96 - 1]

**Never reuses perturbation directions!**

## Multi-GPU Case

With 8 GPUs, 96 perturbations (12 per GPU):

**Step 0:**
- Rank 0: seeds [42, 43, ..., 53] (12 seeds)
- Rank 1: seeds [54, 55, ..., 65]
- ...
- Rank 7: seeds [126, 127, ..., 137]

**Step 1:**
- Rank 0: seeds [138, 139, ..., 149]
- Rank 1: seeds [150, 151, ..., 161]
- ...
- Rank 7: seeds [222, 223, ..., 233]

Each GPU gets a disjoint slice, and seeds evolve across steps.

## Implementation Changes

### 1. Add Step Counter

```python
def __init__(self, ...):
    # ... existing init ...
    self.step_counter = 0  # ADD THIS
```

### 2. Update Seed Calculation

```python
def step(self, ...):
    # OLD (WRONG):
    # start_idx = self.rank * pert_per_worker
    # perturbations = self.generate_perturbations(pert_per_worker, seed_offset=start_idx)

    # NEW (CORRECT):
    step_seed_offset = self.step_counter * self.n_perturbations
    start_idx = self.rank * pert_per_worker

    for i in range(pert_per_worker):
        seed = self.base_seed + step_seed_offset + start_idx + i
        pert = self._generate_single_perturbation(seed)
        # ... use perturbation ...

    self.step_counter += 1  # INCREMENT THIS
```

### 3. Checkpoint Support

```python
def state_dict(self):
    return {
        'learning_rate': self.learning_rate,
        'epsilon': self.epsilon,
        'n_perturbations': self.n_perturbations,
        'step_counter': self.step_counter,  # ADD THIS
        # ... other state ...
    }

def load_state_dict(self, state_dict):
    self.learning_rate = state_dict['learning_rate']
    self.epsilon = state_dict['epsilon']
    self.n_perturbations = state_dict['n_perturbations']
    self.step_counter = state_dict.get('step_counter', 0)  # ADD THIS (default 0 for old checkpoints)
    # ... other state ...
```

## Properties

### ✓ Determinism
- Same `base_seed` → same training trajectory
- Reproducible experiments

### ✓ Ergodicity
- Each step uses different perturbations
- Explores full parameter space over time
- No repeated directions

### ✓ Checkpoint Compatibility
- `step_counter` saved in checkpoint
- Resume uses correct seed sequence
- No duplicate perturbations after resume

### ✓ Multi-GPU Compatible
- Each rank gets disjoint seed slice
- Seeds evolve globally across all ranks

## Example

```python
# Configuration
base_seed = 42
n_pert = 96
n_gpus = 8
pert_per_gpu = 12

# GPU 0, Step 0
step_counter = 0
step_offset = 0 * 96 = 0
rank_offset = 0 * 12 = 0
seeds = [42 + 0 + 0 + i for i in range(12)] = [42, 43, ..., 53]

# GPU 0, Step 1
step_counter = 1
step_offset = 1 * 96 = 96
rank_offset = 0 * 12 = 0
seeds = [42 + 96 + 0 + i for i in range(12)] = [138, 139, ..., 149]

# GPU 7, Step 1
step_counter = 1
step_offset = 1 * 96 = 96
rank_offset = 7 * 12 = 84
seeds = [42 + 96 + 84 + i for i in range(12)] = [222, 223, ..., 233]
```

## Files to Update

All zero-order optimizers need this fix:
1. `zero_order_vmap.py`
2. `zero_order_triton.py`
3. `zero_order_batched.py`
4. `zero_order_parallel.py`
5. `zero_order_optimizer.py` (sequential)

## Critical for Training

Without evolving seeds:
- ❌ Optimization gets stuck sampling same directions
- ❌ Gradient estimates have high variance
- ❌ Poor convergence

With evolving seeds:
- ✓ Explores full parameter space
- ✓ Gradient estimates improve over time
- ✓ Better convergence (as step_counter grows, we've sampled more directions)

## Summary

**Before:** Same seeds every step → stuck in same subspace
**After:** Evolving seeds across steps → ergodic exploration

**Critical for zero-order optimization to work!**

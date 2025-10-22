# Parallel Zero-Order Optimization Implementation Plan

## Current Problem
Sequential perturbation evaluation → 192 forward passes take ~100-180 seconds per step!
- GPU utilization: TERRIBLE (idle 99% of the time waiting for parameter manipulation)
- Memory pressure from torch.compile CUDA graphs accumulation
- Performance degrades continuously until OOM

## Solution: Parallel Perturbation Evaluation with vmap

### Architecture

**Current (BROKEN)**:
```python
for i in range(96):
    perturb_model(+ε)  # Modify weights in-place
    loss_plus[i] = model.forward()  # Sequential!
    restore_model()
    perturb_model(-ε)
    loss_minus[i] = model.forward()  # Sequential!
    restore_model()
# Total: 192 sequential forward passes
```

**Correct (Parallel)**:
```python
# Prepare all 192 perturbed parameter sets
all_perturbed_params = []
for i in range(96):
    params_plus = base_params + epsilon * perturbation[i]
    params_minus = base_params - epsilon * perturbation[i]
    all_perturbed_params.extend([params_plus, params_minus])

# vmap over perturbation dimension - ALL PARALLEL!
def forward_with_params(params):
    return functional_call(model, params, (batch_data,))

all_outputs = vmap(forward_with_params)(all_perturbed_params)
# Shape: [192, batch, seq_len, vocab_size]

# Aggregate: compute loss for each perturbation
all_losses = cross_entropy_per_perturbation(all_outputs, targets)
# Shape: [192]

# Split into plus/minus
losses_plus = all_losses[::2]   # [96]
losses_minus = all_losses[1::2]  # [96]
```

### Key Benefits
1. **192× parallelism**: All forward passes run simultaneously
2. **Perfect GPU utilization**: Batched computation keeps GPU saturated
3. **No torch.compile memory leak**: Single batched call instead of 192 sequential calls
4. **Expected speedup**: 50-100× faster (from ~150s/step to ~1-3s/step)

### Implementation Steps

#### 1. Convert Model to Functional API
```python
from torch.func import functional_call, vmap

# Get model parameters as dict
base_params = dict(model.named_parameters())

# Functional forward pass
def functional_forward(params, x):
    return functional_call(model, params, (x,))
```

#### 2. Create Batched Perturbed Parameters
```python
def create_all_perturbed_params(base_params, perturbations, epsilon):
    """
    Args:
        base_params: dict of {name: tensor}
        perturbations: [n_pert, total_params] Rademacher vectors
        epsilon: perturbation scale

    Returns:
        List of 2*n_pert parameter dicts (for +ε and -ε)
    """
    all_params = []
    for i in range(len(perturbations)):
        # Split perturbation into per-parameter pieces
        param_dict_plus = {}
        param_dict_minus = {}
        offset = 0

        for name, param in base_params.items():
            numel = param.numel()
            pert_flat = perturbations[i, offset:offset+numel]
            pert_shaped = pert_flat.reshape(param.shape)

            param_dict_plus[name] = param + epsilon * pert_shaped
            param_dict_minus[name] = param - epsilon * pert_shaped
            offset += numel

        all_params.append(param_dict_plus)
        all_params.append(param_dict_minus)

    return all_params  # Length: 2*n_pert
```

#### 3. Vectorized Forward Pass with vmap
```python
def parallel_perturbed_forward(model, base_params, batch_data, perturbations, epsilon):
    """
    Run all perturbed forward passes in parallel.

    Returns:
        all_logits: [2*n_pert, batch, seq_len, vocab_size]
    """
    all_params = create_all_perturbed_params(base_params, perturbations, epsilon)

    # vmap over parameter sets
    def forward_single(params):
        return functional_call(model, params, (batch_data,))

    # Stack parameters for vmap
    # This is the tricky part - vmap needs consistent structure
    # May need torch.func.stack_module_state

    all_logits = vmap(forward_single)(all_params)
    return all_logits
```

#### 4. Loss Computation
```python
def compute_losses_parallel(all_logits, targets):
    """
    Args:
        all_logits: [2*n_pert, batch, seq_len, vocab_size]
        targets: [batch, seq_len]

    Returns:
        losses: [2*n_pert]
    """
    # Reshape for loss computation
    n_pert = all_logits.shape[0]
    batch, seq_len, vocab = all_logits.shape[1:]

    # Compute cross-entropy for each perturbation
    # logits: [2*n_pert, batch, seq_len, vocab]
    # targets: [batch, seq_len] -> expand to [2*n_pert, batch, seq_len]
    targets_exp = targets.unsqueeze(0).expand(n_pert, -1, -1)

    # Flatten and compute
    logits_flat = all_logits.reshape(-1, vocab)
    targets_flat = targets_exp.reshape(-1)

    losses_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    losses = losses_flat.reshape(n_pert, -1).mean(dim=1)

    return losses
```

#### 5. Gradient Estimation (unchanged)
```python
losses_plus = all_losses[::2]
losses_minus = all_losses[1::2]

for i in range(n_pert):
    grad = (losses_plus[i] - losses_minus[i]) / (2 * epsilon) * perturbation[i]
    grad_buffer += grad
```

### Memory Considerations

**Memory scaling**:
- Current: O(batch * seq_len * vocab) per forward pass
- Parallel: O(n_pert * batch * seq_len * vocab) for all forward passes

With n_pert=96, batch=128, seq_len=64, vocab=100K:
- Logits memory: 96 * 128 * 64 * 100K * 2 bytes = ~15GB

**Solution**: Process in mini-batches of perturbations:
```python
n_pert_per_batch = 16  # Process 16 perturbations at once
for i in range(0, 192, n_pert_per_batch):
    batch_logits = parallel_forward(perturbations[i:i+n_pert_per_batch])
    batch_losses = compute_losses(batch_logits)
    all_losses.extend(batch_losses)
```

This gives 16× parallelism (still massive improvement over sequential) with only ~2.5GB logit memory.

### Expected Performance

**Current**:
- bs=4, cs=2048: ~100-150s per step (192 sequential forward passes)
- ~520ms per forward pass (sequential overhead)

**With Parallel (16× batching)**:
- 12 batches of 16 perturbations
- Each batch: ~16× faster than 16 sequential (perfect batching)
- Per batch: ~520ms for 16 forward passes (vs 8320ms sequential)
- Total: 12 * 520ms = ~6.2s per step

**Speedup**: 150s → 6s = **24× faster!**

**With Full Parallel (192 at once, if memory allows)**:
- Single batched forward: ~1-2s for all 192
- **75-150× faster!**

### Implementation Priority

1. ✅ Document architecture
2. Create functional forward wrapper
3. Implement parameter perturbation batching
4. Add vmap-based parallel evaluation
5. Handle logits aggregation and loss computation
6. Test with small n_pert (16) first
7. Scale up to full 96 perturbations
8. Benchmark and compare to sequential

### Next Steps

Create `zero_order_parallel.py` with clean implementation following this architecture.

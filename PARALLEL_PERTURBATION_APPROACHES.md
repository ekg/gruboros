# Parallel Perturbation Approaches for Zero-Order Optimization

## Goal
Process N perturbations in parallel to achieve ~N× speedup for zero-order optimization at 500M+ parameter scale.

## Fundamental Challenge

**What GPUs support naturally:**
```
Y = X @ W
X: [batch, features] - DIFFERENT data per batch element
W: [features, output] - SAME weights for all batch elements
```

**What we need:**
```
Y[i] = X @ (W + ε·P[i])
X: SAME data (replicated)
P[i]: DIFFERENT perturbation per batch element
```

GPUs are designed for **data parallelism** (same model, different data), not **model parallelism** (different models, same data).

---

## Approaches Tried

### 1. Serial Virtual Perturbations (BASELINE)
**File:** `zero_order_true_virtual.py`

**How it works:**
- Process perturbations sequentially in a for loop
- Each uses virtual generation (on-the-fly from seeds via Triton kernel)
- Memory: O(1) per perturbation (just seeds)

**Performance (80M model, 8 perts):**
- Time: 4.37s
- Time/pert: 0.55s
- Memory: 32 bytes for seeds

**Pros:**
- ✅ Minimal memory footprint
- ✅ Simple and reliable
- ✅ GPU fully saturated during each matmul
- ✅ Works with all model architectures

**Cons:**
- ❌ Sequential execution (no parallelism)

**Status:** WORKING, CURRENT BEST

---

### 2. Triton Kernel with 3D Tensor Broadcasting
**File:** `zero_order_virtual.py` (batched kernel)

**How it works:**
- Create 3D tensor `P[M, K, N]` inside Triton kernel for per-row perturbations
- Use broadcasting: `Y = X@W + ε·(X[:,:,None] * P).sum(axis=1)`
- Each batch row uses different perturbation from 3D tensor

**Performance (80M model, 8 perts):**
- Time: 26.37s (with pbs=8)
- Time/pert: 3.30s
- Speedup: **0.17× (6× SLOWER!)**

**Pros:**
- ✅ Mathematically correct
- ✅ Compiles and runs
- ✅ Results identical to serial (max diff: 9.88e-05)

**Cons:**
- ❌ 3D tensor overhead: [BLOCK_M, BLOCK_K, BLOCK_N] = huge memory/compute per block
- ❌ 6× slower than serial due to tensor expansion overhead
- ❌ Replicated data processing (N×batch_size vs batch_size)

**Status:** WORKING BUT INEFFICIENT

---

### 3. Micro-Batch Materialization
**File:** `zero_order_prng_parallel.py`

**How it works:**
- Materialize perturbations for small micro-batches (8-16 at a time)
- Process micro-batch sequentially but keep memory bounded
- Free memory after each micro-batch

**Performance (80M model, 8 perts in 1 micro-batch):**
- Time: 2.60s
- Time/pert: 0.325s
- Speedup: **2× vs serial**

**Memory:**
- Micro-batch of 8: 8 × 80M × 4 bytes = 2.5 GB
- Much better than 192 GB for full model

**Pros:**
- ✅ 2× speedup achieved
- ✅ Bounded memory usage
- ✅ Simple implementation

**Cons:**
- ❌ Still materializes full perturbation vectors (not virtual)
- ❌ Only 2× speedup, not 8×
- ❌ Memory scales with micro-batch size × model size

**Status:** WORKING, PARTIAL SPEEDUP

---

## Approaches To Try

### 4. Layer-Wise Materialization (NEW)
**File:** `zero_order_layerwise.py` (CREATED, NOT TESTED)

**Concept:**
- Materialize perturbations ONE LAYER AT A TIME
- For layer with weights [K, N], generate perturbations [N_perts, K, N]
- Use standard batched operations within layer
- Free memory after layer completes

**Memory calculation (500M model, 8 perts):**
- Largest layer: ~2K × 8K = 16M params
- 8 perts: 8 × 16M × 4 bytes = **512 MB per layer**
- Much better than 192 GB for full model!

**Pros:**
- ✅ Bounded memory per layer (not per model)
- ✅ Can use standard batched matmul within layer
- ✅ Natural fit for layer-by-layer execution

**Cons:**
- ❓ Unknown speedup (not tested yet)
- ❓ May still have overhead from data replication

**Status:** IMPLEMENTED, NEEDS TESTING

**Key optimization:** Parallelize perturbation generation across layers

---

### 5. On-the-Fly Generation with Efficient Reduction (FUTURE)
**Concept:**
```python
# In Triton kernel:
y_base = tl.dot(X, W)  # Standard optimized matmul

# For perturbation term, custom reduction:
for n in BLOCK_N:
    for m in BLOCK_M:
        pert_contrib = 0
        for k in BLOCK_K:
            pert_val = hash(seed[m], k, n)  # Generate on-demand, no storage
            pert_contrib += X[m, k] * pert_val
        Y[m, n] += ε * pert_contrib
```

**Pros:**
- ✅ Zero materialization (true virtual perturbations)
- ✅ O(1) memory per perturbation

**Cons:**
- ❌ Nested loops look scary/inefficient
- ❌ May not be faster than optimized matmul
- ❌ Complex implementation

**Status:** CONCEPT ONLY, DEFERRED

---

### 6. Selector Matrix Approach (FUTURE)
**Concept:**
```
Y = X @ W + ε · (X @ P @ S)
```
Where:
- P: [K, N×M] - all M perturbations stacked horizontally
- S: [N×M, N] - selector matrix that picks columns for each batch element

**Pros:**
- ✅ Uses standard matmul operations
- ✅ No 3D tensors

**Cons:**
- ❓ Unclear if selector matrix overhead is better than current approaches
- ❓ P matrix is large: K × (N×M)

**Status:** CONCEPT ONLY, NOT IMPLEMENTED

---

## Performance Summary

| Approach | Time (8 perts) | Time/pert | Speedup vs Serial | Memory |
|----------|---------------|-----------|-------------------|---------|
| Serial Virtual | 4.37s | 0.55s | 1.0× | 32 bytes |
| 3D Tensor Triton | 26.37s | 3.30s | 0.17× | Large |
| Micro-Batch (8) | 2.60s | 0.33s | **1.7×** | 2.5 GB |
| Layer-Wise | ? | ? | ? | ~512 MB/layer |

---

## Key Insights

1. **GPU Architecture Limitation:** GPUs don't natively support "different weights per batch element" efficiently

2. **Serial is Already Good:** The GPU is fully saturated during each matmul in serial execution

3. **3D Tensors Are Expensive:** Creating [M, K, N] tensors kills performance due to memory/compute overhead

4. **Memory Hierarchy Matters:**
   - Full model: 8 × 500M × 4 = 192 GB ❌
   - Per layer: 8 × 16M × 4 = 512 MB ✅
   - Virtual (seeds): 8 × 4 = 32 bytes ✅✅

5. **Materialization Trade-off:**
   - Full virtual (serial): Minimal memory, no parallelism
   - Full materialized: Maximum parallelism, prohibitive memory
   - Layer-wise: Balance between memory and parallelism

---

## Recommended Next Steps

### Immediate: Test Layer-Wise Materialization
1. Run `zero_order_layerwise.py` and measure performance
2. Compare against serial baseline
3. Profile memory usage per layer

### If Layer-Wise Works:
- Optimize perturbation generation (parallel across layers?)
- Test at 500M scale
- Integrate into main training loop

### If Layer-Wise Doesn't Help:
- Accept serial virtual perturbations as optimal
- Focus on other optimizations:
  - Reduce number of perturbations (96 → 48)
  - Increase data batch size
  - Use torch.compile() on model
  - Mixed precision (FP16/BF16)

---

## Decision Matrix

**When to use each approach:**

| Criterion | Serial Virtual | Layer-Wise | Micro-Batch |
|-----------|---------------|------------|-------------|
| Model size < 100M | ✅ Best | Overkill | Overkill |
| Model size 100M-1B | ✅ Good | 🧪 Test | OK |
| Model size > 1B | ✅ Safe | 🧪 Promising | Memory issues |
| Memory constrained | ✅ Best | ✅ Good | ❌ Risk |
| Want 2× speedup | ❌ | 🧪 Maybe | ✅ Yes |
| Want 8× speedup | ❌ | ❓ Unknown | ❌ No |

**Legend:** ✅ Recommended | ❌ Not recommended | 🧪 Experimental | ❓ Unknown

---

## Conclusion

We have working implementations with different trade-offs. The serial virtual perturbation approach is solid and memory-efficient. Layer-wise materialization is the most promising next step for achieving speedup without prohibitive memory costs.

The fundamental GPU architecture limitation means we likely won't achieve 8× speedup, but 2-4× seems feasible with layer-wise materialization.

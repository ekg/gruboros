# PersistentGRU Scaling Lesson

## Educational Summary: Why Persistent-over-Time Kernels Fail at Large H

### Performance Results

| Configuration | PersistentGRU | HybridGRU | Ratio |
|---------------|---------------|-----------|-------|
| **Small (B=8, T=512, H=512)** | 104.6K tok/s | 109.0K tok/s | **96%** ✓ |
| **Production (B=90, T=512, H=2048)** | 28K tok/s | 109K tok/s | **26%** ✗ |

### Root Cause: Architectural Mismatch

**PersistentGRU approach (mingru/persistent_gru.py:19-183):**
```python
# Single kernel launch processes all T timesteps
# Grid: (B, ceil_div(H, BLOCK_H))
# Each program handles one (batch, H-tile) across ALL timesteps
for t in range(T):
    for k0 in range(0, H, BLOCK_K):  # Serial K-loop!
        # Load h_k[BLOCK_K] and U3[BLOCK_K, BLOCK_H]
        # Accumulate: acc += h_k * U3
```

**Problem:** The K-loop is **serialized** within each program (16 iterations for H=2048, BLOCK_K=128).
- Low arithmetic intensity: Loads U3[H×3H] repeatedly across tiles
- Custom GEMV (broadcast multiply + sum) can't match cuBLAS tensor-core GEMMs
- Per-program serialization dominates at large H

**HybridGRU approach (mingru/hybrid_fused_gru.py):**
```python
# Launch optimized kernel per timestep
for t in range(T):
    gates = h @ U3  # Single cuBLAS GEMM: (B×H) · (H×3H)
    # Fully parallelized across B, H, K dimensions
    # Uses tensor cores, perfect tiling, high reuse

    # Triton fused cell for pointwise ops
    r, z, n = fused_gru_cell_kernel(gates, i_r, i_z, i_n)
    h = (1-z) * h_prev + z * n
```

**Why it wins:**
- cuBLAS GEMM is **fully parallelized** across all dimensions
- Tensor core acceleration (bf16 → fp32 accumulate)
- High arithmetic intensity: Perfect tiling and reuse
- Per-timestep launch overhead (512 launches) < persistent's serialization cost

### Roofline Analysis

**Recurrent cost per timestep:** 3B·H² MACs

For production config (B=90, H=2048):
- 3 × 90 × 2048² ≈ **1.13 TFLOP** per timestep
- 512 timesteps × 1.13 TFLOP = **579 TFLOP** total

**PersistentGRU:**
- Arithmetic intensity: LOW (serialized K-loop, repeated U3 loads)
- 579 TFLOP / 28K tok/s = **20.7 ms** per token
- Bottleneck: Compute-bound but **inefficient** compute

**HybridGRU:**
- Arithmetic intensity: HIGH (cuBLAS optimal tiling)
- 579 TFLOP / 109K tok/s = **5.3 ms** per token
- Bottleneck: Peak cuBLAS throughput

### When to Use PersistentGRU

**Only for:**
1. **Small H ≤ 768–1024** where K-loop serialization is tolerable
2. **Tiny B** where cuBLAS can't saturate SMs
3. **Single-token inference** where launch overhead dominates

**For training (B=90, H=2048):** Use HybridGRU or StandardGRU (cuDNN)

### Key Takeaway

> "The 'naive' per-timestep approach with optimized library calls beats the 'clever' persistent kernel by 3.9× at production scale."

**Educational lesson:** Reducing kernel launches is only beneficial if the fused kernel remains efficient. At large H, the persistent-over-T serialization creates a **slower** kernel that outweighs launch savings.

### Recommended Optimizations for HybridGRU

Instead of persistent kernels, focus on:

1. **GRU with projection:** H_rec < H_out (e.g., 1024 vs 2048) reduces recurrent matmul to ×4 less flops
2. **Low-rank U factorization:** U3 ≈ A(H×R) · B(R×3H) with R≪H (256-512)
3. **Keep U3 contiguous:** Single [H, 3H] tensor for coalesced loads
4. **BF16 end-to-end:** Keep accum fp32 in GEMM and Triton cell
5. **Fused cell remains optimal:** Pointwise ops (sig/tanh/update) in single Triton kernel

These optimizations work **with** cuBLAS, not against it.

### Files

- `mingru/persistent_gru.py` - Persistent-over-time implementation (keep for reference/small-H)
- `mingru/hybrid_fused_gru.py` - Production implementation (109K tok/s at B=90, H=2048)
- `test_persistent_gru.py` - Benchmark showing scaling behavior
- `train.persistent.sh` - Training script (demonstrates 28K tok/s throughput)

### Conclusion

PersistentGRU taught us an important lesson about GPU kernel design: **library calls for heavy compute (GEMM) + custom kernels for light compute (pointwise ops)** beats "fuse everything" approaches at production scale.

The document boundary fix and cuDNN gate math optimizations remain active and important for correctness and performance!

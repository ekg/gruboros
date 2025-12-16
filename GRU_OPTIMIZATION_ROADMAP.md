# GRU Optimization Roadmap

## Goal: Push throughput beyond 109K tok/s at production scale (B=90, T=512, H=2048)

## Optimization Strategies

### Option 0: Baseline (HybridGRU)
**Current performance:** 109K tok/s

**Architecture:**
```
Input:  [B, T, D=2048]
  ↓ input_projection: Linear(D, 3H)
Gates:  [B, T, 3H=6144]
  ↓ per-timestep: h @ U3[H, 3H]  ← 3B·H² = 1.13 TFLOP/step
Hidden: [B, T, H=2048]
  ↓ output_projection: Linear(H, D)
Output: [B, T, D=2048]
```

**FLOPs per timestep:** 3 × 90 × 2048² ≈ 1.13 TFLOP
**Total (T=512):** 579 TFLOP

---

### Option 1: Projected GRU
**Expected improvement:** 2-4× faster recurrent step (4× less FLOPs)

**Architecture:**
```
Input:  [B, T, D=2048]
  ↓ project_down: Linear(D, H_rec)
Recurrent: [B, T, H_rec=1024]  ← Half the hidden size!
  ↓ input_projection: Linear(H_rec, 3*H_rec)
  ↓ per-timestep: h @ U3[H_rec, 3*H_rec]  ← 3B·H_rec² = 0.28 TFLOP/step (4× less!)
Hidden: [B, T, H_rec=1024]
  ↓ project_up: Linear(H_rec, D)
Output: [B, T, D=2048]
```

**FLOPs per timestep:** 3 × 90 × 1024² ≈ 0.28 TFLOP (4× reduction!)
**Total (T=512):** 145 TFLOP (4× less than baseline)

**Implementation:**
- Wrap existing HybridGRU with projection layers
- No new kernels needed - just smaller matmuls
- Can reuse all existing cuBLAS/Triton code

**Tuning knobs:**
- `H_rec`: Try 1024, 1280, 1536 (trade-off: speed vs capacity)
- Projection placement: Before GRU (down) and after GRU (up)

---

### Option 2: Low-Rank U Factorization
**Expected improvement:** 1.5-2× faster recurrent step (depends on R)

**Architecture:**
```
Input:  [B, T, D=2048]
  ↓ input_projection: Linear(D, 3H)
Gates:  [B, T, 3H=6144]
  ↓ per-timestep: h @ A[H, R] @ B[R, 3H]  ← Two smaller matmuls
        = (h @ A) @ B
        = (B×H)·(H×R) + (B×R)·(R×3H)
        = B·H·R + B·R·3H = B·R·(H + 3H) = 4B·H·R FLOPs
Hidden: [B, T, H=2048]
  ↓ output_projection: Linear(H, D)
Output: [B, T, D=2048]
```

**FLOPs per timestep:**
- Original: 3B·H² = 3 × 90 × 2048² = 1.13 TFLOP
- Factored: 4B·H·R
  - R=256: 4 × 90 × 2048 × 256 = 0.19 TFLOP (6× less, but accuracy loss?)
  - R=384: 4 × 90 × 2048 × 384 = 0.28 TFLOP (4× less)
  - R=512: 4 × 90 × 2048 × 512 = 0.38 TFLOP (3× less)

**Implementation:**
- Factor U3[H, 3H] → A[H, R] · B[R, 3H]
- Two matmuls per timestep: `temp = h @ A; gates = temp @ B`
- No new kernels - just call cuBLAS twice
- Trade-off: Speed vs accuracy (R controls approximation quality)

**Tuning knobs:**
- `R`: Try 256, 384, 512 (lower R = faster but more lossy)
- Initialization: SVD of full-rank U3, or random init and train from scratch

---

### Option 3: Projected + Low-Rank (Combined)
**Expected improvement:** 8-16× faster recurrent step

**Architecture:**
```
Input:  [B, T, D=2048]
  ↓ project_down: Linear(D, H_rec=1024)
  ↓ input_projection: Linear(H_rec, 3*H_rec)
  ↓ per-timestep: h @ A[H_rec, R] @ B[R, 3*H_rec]
        FLOPs = 4B·H_rec·R
        R=256: 4 × 90 × 1024 × 256 = 0.09 TFLOP (12× less!)
  ↓ project_up: Linear(H_rec, D)
Output: [B, T, D=2048]
```

**FLOPs per timestep:** 0.09 TFLOP (12× reduction)
**Total (T=512):** 48 TFLOP vs 579 TFLOP baseline

---

## Testing Strategy

### Phase 1: Benchmark (Isolated, No Training)
**Goal:** Measure forward-pass throughput for each variant

1. **Baseline:** HybridGRU(H=2048) → should match 109K tok/s
2. **Projected:** ProjectedGRU(H_rec=1024, D=2048) → target: 150-200K tok/s
3. **Low-rank:** LowRankGRU(H=2048, R=384) → target: 140-180K tok/s
4. **Combined:** ProjectedLowRankGRU(H_rec=1024, R=256) → target: 200-300K tok/s

**Script:** `benchmark_gru_variants.py`
- Production config: B=90, T=512, D=2048
- Run each variant 100 times, report mean throughput
- Measure GPU memory usage
- Profile with `torch.profiler` to identify bottlenecks

### Phase 2: Training (Full Pipeline)
**Goal:** Verify variants train correctly and converge to same loss

1. Train each variant for 1000 steps on same data
2. Compare:
   - Throughput (tok/s)
   - Loss trajectory
   - GPU memory usage
   - Checkpoint size
3. Pick best variant for full training run

**Scripts:**
- `train.projected_gru.sh` - ProjectedGRU
- `train.lowrank_gru.sh` - LowRankGRU
- `train.combined_gru.sh` - ProjectedLowRankGRU

---

## Code Structure

### Modular Design (No Kernel Changes!)

```
mingru/
  hybrid_fused_gru.py        # Baseline (existing, 109K tok/s)
  projected_gru.py           # Option 1: Projection wrapper
  lowrank_gru.py             # Option 2: Low-rank factorization
  projected_lowrank_gru.py   # Option 3: Both optimizations
```

**Key insight:** All variants use the **same Triton fused cell kernel** from HybridGRU. Only the matmul shapes change:

- **HybridGRU:** `h[B, H] @ U3[H, 3H]` (one big GEMM)
- **ProjectedGRU:** `h[B, H_rec] @ U3[H_rec, 3*H_rec]` (smaller GEMM)
- **LowRankGRU:** `h[B, H] @ A[H, R]` then `temp[B, R] @ B[R, 3H]` (two smaller GEMMs)
- **Projected+LowRank:** `h[B, H_rec] @ A[H_rec, R]` then `temp[B, R] @ B[R, 3*H_rec]` (two tiny GEMMs)

### Integration with train.py

Add arguments:
```python
--gru_variant {hybrid, projected, lowrank, combined}
--h_recurrent 1024  # For projected variants
--rank 384          # For low-rank variants
```

---

## Deliverables

### Milestone 1: Benchmark Script (Today)
- [ ] `benchmark_gru_variants.py` - Compare all 4 variants
- [ ] Report: Throughput, memory, profile for each

### Milestone 2: Implementation (Next)
- [ ] `mingru/projected_gru.py` - Clean wrapper around HybridGRU
- [ ] `mingru/lowrank_gru.py` - Factored U3 variant
- [ ] Verify correctness: Forward pass matches baseline (within tolerance)

### Milestone 3: Training Test (After)
- [ ] Train each variant for 1000 steps
- [ ] Compare loss curves
- [ ] Pick winner for production

---

## Expected Outcomes

| Variant | FLOPs/step | Expected tok/s | Memory | Accuracy |
|---------|------------|----------------|--------|----------|
| Baseline | 1.13 TFLOP | 109K ✓ | 32 GB | 100% |
| Projected (H_rec=1024) | 0.28 TFLOP | 150-200K | 24 GB | 95-98% |
| Low-rank (R=384) | 0.28 TFLOP | 140-180K | 28 GB | 90-95% |
| Combined (H_rec=1024, R=256) | 0.09 TFLOP | 200-300K | 18 GB | 85-92% |

**Trade-off:** Higher throughput ↔ Slightly lower capacity

**Strategy:** Start with Projected (cleanest, best accuracy/speed), then add low-rank if needed.

---

## Next Steps

1. ✅ Document roadmap (this file)
2. ⏭️ Create `benchmark_gru_variants.py` skeleton
3. ⏭️ Implement `ProjectedGRU` (simplest, highest ROI)
4. ⏭️ Benchmark to validate 2× speedup
5. ⏭️ If successful, implement `LowRankGRU`
6. ⏭️ If both successful, combine for maximum speedup

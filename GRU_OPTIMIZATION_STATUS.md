# GRU Optimization Progress

## Status: Phase 1 Complete (Benchmark Infrastructure + ProjectedGRU)

### Completed ✓

1. **Roadmap created** (`GRU_OPTIMIZATION_ROADMAP.md`)
   - Documented 4 optimization strategies
   - FLOPs analysis for each approach
   - Expected performance targets

2. **Benchmark infrastructure** (`benchmark_gru_variants.py`)
   - Modular design for testing variants
   - Production config: B=90, T=512, D=2048
   - Measures: throughput, latency, memory

3. **ProjectedGRU implemented** (`mingru/projected_gru.py`)
   - Wraps HybridFusedGRU with down/up projections
   - Tested H_rec sweep: 768, 1024, 1280, 1536

### Results

| Variant | H_rec | Throughput | Speedup | Params | Memory |
|---------|-------|------------|---------|--------|--------|
| Baseline | 2048 | 385K tok/s | 1.00× | 25.2M | 0.41 GB |
| Projected | 768 | 477K tok/s | 1.24× | 6.7M | - |
| **Projected** | **1280** | **489K tok/s** | **1.27×** | **15.1M** | **-** |
| Projected | 1024 | 431K tok/s | 1.12× | 10.5M | 0.38 GB |
| Projected | 1536 | 403K tok/s | 1.05× | 20.5M | - |

**Key finding:** H_rec=1280 gives best throughput (1.27× speedup), but speedup is modest due to projection overhead.

### Analysis

**Why only 1.27× speedup (not 2-4× as predicted)?**

Theoretical FLOPs reduction (H_rec=1280 vs H=2048):
- Recurrent matmul: 3B·H² → 3B·H_rec²
- Reduction: (1280/2048)² = 0.39 (2.6× less FLOPs)

But total forward pass includes:
1. **Input projection:** D → 3H_rec (new overhead!)
2. **Recurrent matmul:** H_rec @ U3[H_rec, 3H_rec] (this gets faster)
3. **Output projection:** H_rec → D (new overhead!)
4. **Fused cell:** Pointwise ops (unchanged)

The projection layers (1 & 3) add overhead that partially cancels the recurrent savings (2).

**Recommendation:** Try low-rank factorization next - it reduces recurrent matmul without adding projection layers.

---

## Next: Phase 2 - LowRankGRU

### Goal
Factor U3[H, 3H] → A[H, R] · B[R, 3H] to reduce recurrent FLOPs without projection overhead.

### Expected Performance
- **R=384:** ~140-180K tok/s (target: 1.4-1.8× baseline)
- **R=512:** ~130-160K tok/s (target: 1.3-1.6× baseline)

**Hypothesis:** Low-rank will beat ProjectedGRU because it avoids extra projection layers.

### Implementation Plan

1. Create `mingru/lowrank_gru.py`
   - Factor U3 into A·B
   - Forward: `gates = (h @ A) @ B` (two matmuls)
   - Keep same input/output dimensions (no projections!)

2. Benchmark sweep over R: 256, 384, 512

3. Compare: LowRankGRU vs ProjectedGRU vs Baseline

4. If LowRankGRU wins, implement combined ProjectedLowRankGRU

---

## Deliverables

- [x] Roadmap document
- [x] Benchmark infrastructure
- [x] ProjectedGRU implementation
- [x] H_rec parameter sweep
- [ ] LowRankGRU implementation
- [ ] Rank parameter sweep
- [ ] Combined ProjectedLowRankGRU (if both work)
- [ ] Training validation (1000 steps)
- [ ] Production training with winner

---

## Files

- `GRU_OPTIMIZATION_ROADMAP.md` - Strategy and FLOPs analysis
- `GRU_OPTIMIZATION_STATUS.md` - This file (progress tracker)
- `benchmark_gru_variants.py` - Benchmark script
- `mingru/projected_gru.py` - ProjectedGRU implementation
- `mingru/lowrank_gru.py` - TODO: Next implementation
- `sweep_h_rec.sh` - H_rec parameter sweep script


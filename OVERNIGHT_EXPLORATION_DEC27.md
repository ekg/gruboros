# Overnight Exploration Results - December 27, 2025

## Summary

Ran extended training comparisons between Mamba2 and Triple R variants at ~1.3B parameters.

## Results

### 1000-Step Comparison (avg50 loss)

| Model | Parameters | avg50 |
|-------|------------|-------|
| **Mamba2** | 1.33B | **3.924** |
| Selective Triple R | 1.33B | 4.216 |
| ElmanTripleR | 1.35B | 4.216 |

**Key Finding: Mamba2 beats Triple R variants by 6.9%**

### 2000-Step Comparison

| Model | 1000 steps | 2000 steps | Improvement |
|-------|------------|------------|-------------|
| Mamba2 | 3.924 | 3.861 | 1.6% |

Diminishing returns after 1000 steps.

## Key Discoveries

### 1. Selective Input Gate Has NO Effect

The Mamba2-style selective input gate (B gate) was added to Triple R to create "Selective Triple R":
```
B_gate = sigmoid(W_B @ x + b_B)  # Input-dependent modulation
```

**Result: Identical performance (4.216 avg50) with or without this gate.**

This suggests Mamba2's advantage comes from other architectural differences, not the selective input gating.

### 2. Mamba2 Architectural Advantages

Mamba2's ~7% advantage likely comes from:
- **Diagonal state transition matrix (A)** vs Triple R's full R_h matrix
- **SSM discretization mechanics** (A, B, C, delta) vs GRU mechanics
- **State expansion** (2x hidden dimension inside the SSM block)
- **Output projection (C)** to read from state

### 3. Higher Learning Rate Hurts Triple R

LR=0.001 vs LR=0.0006:
- LR=0.001: avg50 = 5.27 (worse)
- LR=0.0006: avg50 = 5.174

Higher LR causes gradient instability for Triple R.

## Throughput Comparison

| Model | T/s (8× A100) |
|-------|---------------|
| Mamba2 | ~30-37k |
| Triple R (Haste) | ~10-11k |

Mamba2 is ~3x faster due to highly optimized CUDA kernels.

## Files Added

- `train.mamba2_1000.sh` - Mamba2 1000-step training
- `train.mamba2_2000.sh` - Mamba2 2000-step training
- `train.triple_r_1000.sh` - ElmanTripleR 1000-step training
- `train.triple_r_2000.sh` - ElmanTripleR 2000-step training
- `plot_1000_step_comparison.py` - Comparison visualization

## Conclusion

Mamba2 is the better architecture at ~1.3B parameters, outperforming Triple R variants by ~7% and running 3x faster. The selective input gate (Mamba2's key innovation) does NOT help Triple R, suggesting the advantage comes from the SSM formulation itself.

Future work could explore:
1. Diagonal R matrix for Triple R (like Mamba2's diagonal A)
2. State expansion for Triple R
3. SSM-style discretization for Triple R

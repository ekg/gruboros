# Gruboros Research Status

## Executive Summary

We're exploring the design space between **Mamba-style selectivity** and **simple Elman RNNs** to understand what makes recurrent language models work. Key question: Can we match Mamba2/Transformer performance with simpler, more interpretable architectures?

**Latest finding (Dec 27)**: Mamba2 definitively beats Triple R by 7% at 1000 steps (3.924 vs 4.216 avg50). The selective input gate (Mamba2's B gate) has NO effect when added to Triple R. Mamba2's advantage likely comes from:
1. **Log-space numerical stability** - Custom segment-sum algorithm prevents NaN during training
2. **Diagonal state transitions** - Element-wise decay is more stable at depth than full R matrices
3. **SSD parallelization** - Hardware-efficient matrix multiplication primitives

We're now testing **Diagonal MHTR** (Multi-Head Triple R with diagonal transitions) to isolate whether diagonal structure is the key.

---

## Current Results (Dec 27, 2025)

### 1000-Step Extended Comparison (~1.3B params, 8× A100)

| Rank | Model | Params | avg50 | tok/s | Notes |
|------|-------|--------|-------|-------|-------|
| 1 | **Mamba2** | 1.33B | **3.924** | 25-35k | Winner by 7% |
| 2 | MHTR (Multi-Head Triple R) | 1.33B | 4.197 | 15-16k | 32× state expansion |
| 3 | Triple R | 1.28B | 4.216 | 10-11k | Full R matrices |
| 4 | Selective Triple R | 1.33B | 4.216 | 10-11k | +B gate = no effect |

### 300-Step Quick Comparison (previous results)

| Model | Params | avg50 | Notes |
|-------|--------|-------|-------|
| Mamba2 | 1.33B | 5.04-5.13 | mamba-ssm package |
| Triple R | 1.28B | 5.179 | 3 R matrices |
| LLaMA Transformer | 1.34B | 5.685 | RoPE, SwiGLU, RMSNorm |

**Key observation**: Mamba2's lead widens with more training (2.7% at 300 steps → 7% at 1000 steps).

---

## Research Direction

We started from two endpoints:
1. **Mamba2**: Complex selective SSM with input-dependent A, B, C matrices
2. **Simple Elman**: h_t = σ(W_x @ x_t + W_h @ h_{t-1})

Our approach: Systematically add components from Mamba2 to Elman to find the minimal effective architecture.

### What We've Learned

**From Mamba2 (SSM) side:**
- Selectivity (input-dependent gating) is important
- Diagonal A matrices are efficient but sacrifice expressivity
- SiLU activation works well as output gate
- Delta (discretization step) controls memory/update tradeoff

**From Elman (simple RNN) side:**
- Full R matrices (not diagonal) enable richer dynamics
- Competition (softmax over groups) provides implicit sparsity
- The key is getting gradients to flow through long sequences
- Simpler is often better when combined with right inductive biases

**Our synthesis (Triple R):**
- Keep full R matrices (expressivity)
- Add input-dependent delta (selectivity)
- Use compete×silu output gate (gradient flow + sparsity)
- No FFN needed - pure recurrent works at scale

---

## Triple R Architecture Details

**ElmanTripleRCompeteSilu** - Our best model with 3 recurrent weight matrices:

```python
# Dynamics (simplified)
delta = sigmoid(W_delta @ x)           # Input-dependent decay [0, 1]
delta_mod = delta @ R_delta            # Per-dimension delta modulation
h_new = delta * h_prev @ R_hidden      # Decayed recurrence
h_new = h_new + x @ R_input            # Add input contribution
h_new = h_new + delta_mod              # Apply delta modulation

# Output gating
compete = softmax(h_new, groups=32)    # Group competition (implicit sparsity)
output = compete * silu(h_new)         # Compete × SiLU gate
```

**Why 3 R matrices?**
- **R_input** (D×D): Transforms input embedding to hidden state
- **R_hidden** (D×D): Recurrent transformation, hidden-to-hidden
- **R_delta** (D×D): Modulates the discretization step per-dimension

The key insight is that separating these transformations gives the model more flexibility to learn different dynamics for input processing, memory retention, and gating - similar to how LSTM/GRU separate gates.

---

## Architecture Evolution

### Phase 1: Baseline Elman Variants
Started with discretized Elman RNN variants testing different gating mechanisms:

1. **ElmanLeaky** - Basic discretized dynamics with input-dependent delta
2. **ElmanLeakySilu** - Added SiLU activation (like Mamba2)
3. **ElmanLeakyCompete** - Group softmax competition gate (groups of 32 dims)
4. **ElmanLeakyCompeteSilu** - Compete × SiLU hybrid = **Current Baseline (5.223)**

Key insight: Competition gate + SiLU output provides best gradient flow.

### Phase 2: Triple R Architecture
Result: 5.179 vs baseline 5.223 = **0.044 improvement** with more expressivity.

### Phase 3: Ablation Studies

| Variant | Params | avg50 | Status |
|---------|--------|-------|--------|
| Triple R | 1.28B | 5.179 | **Best RNN** |
| Mamba2 (official) | 1.33B | 5.04-5.13 | Reference |
| LLaMA Transformer | 1.34B | 5.685 | Slower convergence |
| Low-Rank R (rank=256) | 1.18B | 5.788 | Failed - rank too aggressive |
| Neural Memory Bank | pending | - | Not yet run |

---

## Blocking Issue: Gradient Death

### Why does gradient death occur after ~350k steps?
**Status: Unsolved - This is the main blocker for long training.**

7 hypotheses tested, none fully solved:
1. Learning rate decay → Tried warmup/cooldown, no fix
2. Hidden state saturation → Checked norms, not the cause
3. Gate collapse → Gates remain active
4. Gradient clipping too aggressive → Varied from 0.5 to 2.0, minimal effect
5. Optimizer state stale → Tried resets, no fix
6. Data distribution shift → Pile data is well-shuffled
7. Numerical precision issues → BF16 vs FP32 similar behavior

**Best mitigation**: Train for fewer steps at higher learning rate.

**This needs investigation before scaling to longer training runs.**

---

## Open Questions

### 1. Is Mamba2's advantage from architecture or parallelization?
**Status: Partially answered**
- Mamba2's SSD (State Space Duality) enables hardware-efficient training
- Our Haste CUDA kernels achieve similar throughput (~20-25k tok/s)
- Architecture-wise: Mamba2 uses diagonal A (no R matrix) + selective gating
- Triple R with full R matrices is competitive with Mamba2

### 2. Does explicit Δ (discretization step) help Elman catch up to GRU?
**Status: Yes**
- Input-dependent delta is crucial for learning rate adaptation per position
- Delta initialization matters: -1.8 to -2.0 works well (sigmoid → ~0.14-0.12 base decay)
- Per-dim delta scaling (R_delta matrix in Triple R) adds further expressivity

### 3. Is SiLU output gating the key ingredient?
**Status: Yes, but competition helps too**
- SiLU alone: Good but not optimal
- Competition gate alone: Sparse but harder to train
- Compete × SiLU hybrid: Best of both worlds
  - Competition provides implicit sparsity
  - SiLU provides smooth gradients

### 4. Why do transformers converge slower in early training?
**Status: Observed but unexplained**
- LLaMA at 300 steps: 5.685 (vs Triple R 5.179, Mamba2 5.04)
- Transformers typically need more tokens to reach same loss
- Hypothesis: Recurrent models have stronger sequential inductive bias

---

## Other Research Directions

### Zero-Order Optimization (CD-RGE)
We're also exploring zero-order (gradient-free) optimization for memory efficiency:
- **Goal**: Train 500M+ param models without storing activations for backprop
- **Approach**: Central-Difference Random Gradient Estimation
- **Key innovation**: Virtual perturbations (generate from seeds, never materialize)
- **Status**: Early experiments with StandardGRU, not yet integrated with Elman variants
- **Files**: `zero_order_optimizer.py`, `zero_order_virtual.py`

### Gossip-Based Evolution
Original gruboros concept: evolutionary optimization across distributed models
- Each GPU runs independent training
- Models exchange parameters based on fitness
- **Status**: Implemented but not actively used for current architecture comparison
- **Files**: `gossip/evolutionary_node.py`, `gossip/network_utils.py`

---

## Backend Selection Guide

| Backend | Speed | DDP | Memory | Use Case |
|---------|-------|-----|--------|----------|
| **Haste CUDA** | Fastest | Yes | Low | Production (ElmanLeaky*, Triple R) |
| cuDNN | Fast | Yes | Medium | StandardGRU reference |
| FlashRNN | Fast | No | Low | Research (some DDP issues) |
| Sequential Triton | Medium | Yes | High | Debugging/validation |
| PyTorch Sequential | Slow | Yes | High | Reference implementation |

**Recommendation**: Use Haste CUDA kernels for all Elman variants.

---

## Training Configuration (Current Standard)

```bash
--dim 2048
--depth 32
--expansion_factor 2.0  # 2x hidden dim
--ff_mult 0.0           # No FFN (pure recurrent)
--chunk_size 512
--batch_size 16
--lr 0.0006
--weight_decay 0.1
--grad_clip 1.0
--compete_n_groups 32
--delta_init -1.8
```

**Token throughput target**: 20-30k tok/s on 8× A100/H100

---

## Files to Know

### Core Models
- `mingru/haste_elman_triple_r.py` - Triple R implementation (Haste CUDA)
- `mingru/haste_elman_compete_silu.py` - Baseline compete×silu
- `mingru/multihead_triple_r_expanded.py` - MHTR with 32× state expansion
- `mingru/diagonal_mhtr.py` - Diagonal MHTR (Mamba2-style transitions)
- `mingru/logspace_hybrid_gru.py` - Log-space GRU (numerical stability)
- `mingru/mamba_lm.py` - Official Mamba2 wrapper
- `mingru/llama_lm.py` - LLaMA transformer baseline

### Training
- `train.py` - Main training script (supports all model types)
- `train.triple_r_compete_silu.sh` - Triple R training
- `train.real_mamba2.sh` - Mamba2 comparison
- `train.llama_1b.sh` - Transformer baseline

### Zero-Order Optimization
- `zero_order_optimizer.py` - CD-RGE optimizer with seed-based perturbations
- `zero_order_virtual.py` - Triton kernel for virtual perturbations

### Gossip Evolution
- `gossip/evolutionary_node.py` - Per-GPU evolutionary logic
- `gossip/network_utils.py` - TCP peer-to-peer communication
- `gossip/fitness_tracker.py` - Loss-based fitness evaluation

---

## Next Steps (Priority Order)

1. **Complete Diagonal MHTR experiment** - Does diagonal match Mamba2?
   - If yes → diagonal transitions are key, not expressivity
   - If no → Mamba2's advantage is log-space + SSD parallelization
2. **Implement log-space Elman** - Port Mamba2's segment-sum to our architecture
3. **Profile numerical stability** - Track hidden state norms/gradients through depth
4. **Investigate gradient death** - May be related to numerical instability
5. **Wallclock comparison** - Fair throughput at max batch size

---

## Key Insights

1. **RNNs converge faster early**: At 300 steps, Triple R (5.18) and Mamba2 (5.04) both beat LLaMA transformer (5.69). Recurrent inductive bias helps with early language modeling.

2. **Mamba2's lead widens over time**: 2.7% gap at 300 steps → 7% at 1000 steps. This suggests Mamba2 has better gradient flow for extended training.

3. **Selective input gate (B) doesn't help**: Adding Mamba2's `B = sigmoid(W_B @ x)` to Triple R had zero effect. Selectivity is not the source of Mamba2's advantage.

4. **Full R matrices may hurt at depth**: Hypothesis: Full matrix transitions `h = R @ h` compound errors through layers. Diagonal transitions `h = R * h` (element-wise) are more stable.

5. **Log-space computation is critical**: Mamba2 uses log-space segment-sum algorithm that "without the right implementation... produces NaNs immediately during training (even with FP32)." Our RNNs may be hitting similar numerical limits.

6. **Competition + SiLU synergy**: Competition gate provides implicit sparsity/selection. SiLU provides smooth gradients. Combined = best of both.

7. **Delta initialization matters**: -1.8 to -2.0 range keeps initial decay ~0.12-0.14, allowing gradients to flow while maintaining memory.

8. **Throughput gap persists**: Mamba2 is ~2-3× faster (25-35k vs 10-16k tok/s) due to SSD using tensor cores for matrix multiplication.

---

## Progress Log

### December 27, 2025

1. **1000-step extended comparison** - Mamba2 beats Triple R by 7% (3.924 vs 4.216)
2. **Selective input gate ablation** - Adding B gate to Triple R had NO effect
3. **Multi-Head Triple R (MHTR)** - 32× state expansion, still loses to Mamba2 (4.197 vs 3.924)
4. **Diagonal MHTR experiment** - Started testing diagonal R transitions (Mamba2-style)
5. **Log-space hypothesis** - Mamba2's segment-sum algorithm may be key to deep training stability

**Key finding**: Mamba2's advantage is NOT from selectivity. It's likely from:
- Log-space numerical stability (prevents NaN at depth)
- Diagonal state transitions (independent per-dimension decay)
- Hardware-efficient SSD algorithm

**Research pivot**: From "can rich R matrices beat SSMs?" to "is diagonal structure + log-space the winning formula?"

### December 26, 2025

1. **Fixed param counting** - Now shows actual model params, not estimates
2. **Ran Mamba2 comparison** - Official mamba-ssm at 1.33B: 5.04-5.13 avg50
3. **Completed LLaMA transformer** - 1.34B: 5.685 avg50 (slower than RNNs)
4. **Created comparison plot** - `/tmp/dec26_model_comparison.png`

---

## Hypothesis: Log-Space Enables Depth

Mamba2's SSD algorithm uses a critical numerical technique:

```
Instead of computing cumulative products (which underflow):
  A_1, A_1·A_2, A_1·A_2·A_3, ...

Convert to log-space cumulative sums:
  log(A_1), log(A_1)+log(A_2), log(A_1)+log(A_2)+log(A_3), ...
```

But naive log-space still fails due to **catastrophic cancellation** when subtracting large cumulative sums. Mamba2 uses a custom **segment-sum (segsum)** operation that "produces the right answer without subtraction."

**Quote from Tri Dao's blog**: "Without the right implementation of these primitives, the basic SSD algorithm produces NaNs immediately during training (even with FP32)."

**Implication for Triple R**: Our full R matrices may accumulate numerical errors through depth. If diagonal MHTR matches Mamba2, the next step is implementing log-space Elman dynamics.

*Last updated: December 27, 2025*

# Discretized Selective RNN Research Plan

## Overview

Investigating whether classical RNN architectures (Elman, GRU), when augmented with:
1. **Explicit input-dependent discretization (Δ)**
2. **Selective output gating**

...can match or exceed modern SSMs (Mamba-2) while retaining full representational expressivity.

**Core hypothesis:** The linear SSM structure in Mamba is a training efficiency compromise, not a representational advantage. By adding Mamba's "selectivity" innovations to nonlinear RNNs, we can achieve competitive performance without expressivity limitations.

---

## Critical Fix Applied (Dec 22, 2025)

### Padding Contamination Issue

**Problem discovered:** All previous loss comparisons were contaminated by padding tokens being included in loss computation. Models appeared to have lower loss because they learned to predict padding (easy pattern).

**Fix applied:** Added `actual_length` parameter to all model forward() methods:
- `CuDNNGRU_MultLM`
- `HasteGRU_MultLM`
- `MambaLM` / `Mamba2LM`
- `Mamba2FFN3LM`
- `HybridMamba2GRULM`

Loss masking uses `ignore_index=-100` to exclude padding from cross-entropy.

**Impact:** Previous "good" results (e.g., 1.49 loss at step 14) were fake. Honest loss with masking is ~4x higher but represents true language modeling performance.

### Honest Baseline Results (Dec 23, 2025)

Config: batch=24, chunk=512, 8 GPUs, ~1B params

| Model | Step 3000 Loss | Tokens/sec |
|-------|----------------|------------|
| HasteGRU_Mult (1.01B) | 4.38 | 25.6k |
| Mamba2 SSD (1.00B) | 3.68 | 35.7k |

**Key finding:** Mamba2 shows ~16% lower loss per step. This is the honest starting point.

---

## Theoretical Framework

### The Discretization Continuum

```
Linear SSM (Mamba):     h_t = α_t h_{t-1} + B_t x_t
Discretized Elman:      h_t = (1-Δ_t) h_{t-1} + Δ_t tanh(Wx + Uh)
GRU:                    h_t = (1-z_t) h_{t-1} + z_t tanh(Wx + U(r⊙h))
```

All three can be viewed as variants of:
```
h_t = (1 - gate_t) h_{t-1} + gate_t · candidate_t
```

The differences are:
- **What determines the gate:** Fixed Δ, input-dependent Δ, input+state-dependent z
- **Candidate nonlinearity:** Linear projection, tanh, tanh with reset gate
- **Gate constraints:** softplus (unbounded), sigmoid (0-1)

### Why Discretized Elman is the Right Starting Point

1. **Minimal architecture** - Isolates the effect of discretization
2. **Clear theory** - Direct analog to continuous-time leaky integrator
3. **Fast implementation** - Simpler than GRU, amenable to fusion
4. **Interpretable Δ** - Can analyze what the model learns about "step size"

---

## Implementation Plan

### Phase 0: Infrastructure (IN PROGRESS)

- [x] Fix padding contamination in all models
- [x] Establish honest Mamba2 baseline with corrected loss
- [x] Establish honest HasteGRU_Mult baseline
- [x] Update plotting scripts for corrected comparisons
- [ ] Implement DiscretizedElman in haste framework
- [ ] Apply actual_length fix to new Elman models
- [ ] Create standardized benchmark config (3k steps, batch=24, chunk=512)

### Phase 1: Discretized Elman Ladder

All experiments with and without selective output layer.

| Experiment | Δ Mode | Description |
|------------|--------|-------------|
| 1.1 | Fixed scalar | Δ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} |
| 1.2 | Learned global | softplus(learned_param) per layer |
| 1.3 | Input-dependent scalar | Δ_t = softplus(w·x_t + b) |
| 1.4 | Input-dependent vector | Δ_t = softplus(W·x_t + b) |
| 1.5 | Input+state dependent | Δ_t = softplus(W_x·x_t + W_h·h_t + b) |

**Target:** Find minimal Δ mode that matches/exceeds Mamba2.

### Phase 2: Selective Output Ablations

| Experiment | Output Mode | Description |
|------------|-------------|-------------|
| 2.1 | None | Direct h output |
| 2.2 | SiLU gate | gate = silu(W_h·h + W_x·x) |
| 2.3 | Sigmoid gate | gate = sigmoid(W_h·h + W_x·x) |
| 2.4 | Input-only gate | gate = silu(W_x·x) |
| 2.5 | State-only gate | gate = silu(W_h·h) |

**Target:** Determine if selectivity is load-bearing and which form works best.

### Phase 3: Architecture Comparisons

| Model | Δ Mode | Selective Output | Params |
|-------|--------|------------------|--------|
| Mamba2 SSD | N/A (native) | N/A (native) | 1.00B |
| DiscretizedElman | Best from Phase 1 | Best from Phase 2 | ~1.00B |
| HasteGRU_Mult | Native z gate | SiLU | 1.01B |
| DiscretizedGRU | Explicit Δ | SiLU | ~1.00B |

**Target:** Fair comparison at matched param count and training compute.

### Phase 4: Multi-Head Variants

If single-head is competitive but slow:
- Test H ∈ {4, 8, 16, 32, 64} heads
- Measure throughput vs quality tradeoff
- Compare to Mamba2's inherent parallelism

### Phase 5: Capability Evaluation

After identifying best architecture:

| Task | Why It Matters |
|------|----------------|
| Parity | Mamba-2 fails, tests nonlinear dynamics |
| Modular arithmetic | Mamba-3 benchmark |
| Associative recall | Memory capacity test |
| Length generalization | Train 512, test 1K-8K |

---

## File Organization

```
mingru/
├── discretized_elman.py      # New: DiscretizedElmanLM
├── discretized_gru.py        # New: DiscretizedGRULM
├── haste_gru_mult.py         # Existing: HasteGRU_MultLM
├── mamba_lm.py               # Existing: Mamba2LM (baseline)
└── ...

experiments/
├── phase1_elman_delta/       # Δ mode sweeps
├── phase2_selective/         # Output ablations
├── phase3_comparison/        # Architecture comparison
└── phase4_multihead/         # Scaling experiments
```

---

## Training Configuration

### Standard Benchmark Config
```bash
--batch_size 24
--chunk_size 512
--train_steps 3000        # Quick iteration
--lr 0.001
--sf_beta 0.9
--sf_beta2 0.995
--weight_decay 0.033
--grad_clip 1.0
--bf16
--no-tbptt               # No hidden state carryover (match Mamba2)
```

### Full Training Config
```bash
--train_steps 10000       # For final comparisons
```

---

## Key Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Loss @ 3k steps | ≤ Mamba2 (3.68) | Primary comparison |
| Final loss @ 10k | Track convergence | Secondary |
| Tokens/sec | Track but not optimize | Can adjust batch size |
| GPU memory | Reasonable | Must fit on 48GB |

---

## Current Status

### Completed
- [x] Padding contamination fix (commit 80962f8)
- [x] Honest Mamba2 baseline: 3.68 loss @ step 3000
- [x] Honest HasteGRU_Mult baseline: 4.38 loss @ step 3000
- [x] R plotting script updated for corrected comparisons

### In Progress
- [ ] Design DiscretizedElman architecture
- [ ] Implement in haste framework
- [ ] Create training scripts

### Blocked
- Nothing currently

---

## Notes & Observations

### Dec 23, 2025
- Discovered padding contamination was making GRU look artificially good
- With honest loss masking, Mamba2 outperforms HasteGRU_Mult by ~16% per step
- This changes the research question: can we close this gap with discretization?

### Open Questions
1. Is the Mamba2 advantage from architecture or from better parallelization of training?
2. Does explicit Δ help Elman catch up to GRU's implicit gating?
3. Is selective output (SiLU gating) the key ingredient or is discretization sufficient?

---

## References

- Mamba: Gu & Dao, 2023, arXiv:2312.00752
- Mamba-2 (SSD): Dao & Gu, 2024, arXiv:2405.21060
- GRU: Cho et al., 2014
- Haste library: https://github.com/lmnt-com/haste

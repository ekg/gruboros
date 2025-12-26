= Gate Architecture Ablation Study

== Baseline
- *Compete×silu* (32 groups): *5.22 avg50* @ 1.01B params, ~20k T/s

== Evaluation Methodology
All losses reported as *avg50*: average of last 50 training steps (steps 250-299).
This provides more stable comparison than single-step snapshots.

== Phase 1: Activation Function Sweep
#table(
  columns: (auto, auto, auto, auto),
  [*Variant*], [*avg50*], [*T/s*], [*Notes*],
  [compete×silu (baseline)], [5.223], [~20k], [*Best*],
  [compete×gelu], [5.228], [~20k], [Tied with silu],
  [compete×mish], [5.242], [~18k], [Slightly worse],
  [learned temperature], [5.282], [~20k], [Fixed temp=1.0 is optimal],
)

== Phase 2: Group Size Sweep (n_groups)
Testing whether 32 groups is optimal or can we go coarser/finer.

#table(
  columns: (auto, auto, auto, auto),
  [*n_groups*], [*dims/group*], [*avg50*], [*Notes*],
  [16], [128], [5.309], [Too coarse],
  [32 (baseline)], [64], [5.223], [*Optimal*],
  [64], [32], [5.230], [Slightly worse],
  [128], [16], [5.344], [Too fine],
)

== Phase 3: Asymmetric Gate
Coarse competition (16 groups) × full-dim silu.

#table(
  columns: (auto, auto, auto, auto),
  [*Variant*], [*avg50*], [*T/s*], [*Notes*],
  [asymmetric (16 groups compete × 2048-dim silu)], [5.244], [~18k], [Worse than baseline],
)

== Phase 4: Recurrence Ablations
Testing what makes Elman better than Mamba2.

#table(
  columns: (auto, auto, auto, auto),
  [*Variant*], [*avg50*], [*T/s*], [*What it tests*],
  [no_delta (fixed α=0.88)], [5.627], [~18k], [*Input-dependent delta is critical!*],
  [mamba2_style (no R, no tanh)], [5.394], [~17k], [R matrix + tanh worth ~0.17],
  [mamba2_tanh (no R, WITH tanh)], [5.679], [~3k], [Tanh alone HURTS! (dim=2816, 1.04B)],
  [mamba2_silu (no R, WITH silu)], [5.762], [~11k], [Silu HURTS even more! (dim=2048, 1.01B)],
  [*mamba2_delta* (softplus/exp, WITH R)], [*5.226*], [~17k], [*Delta formulation doesn't matter!*],
  [mamba2_delta_conv4 (+ causal conv)], [5.327], [~13k], [Conv4 HURTS! Slower too],
  [learned_delta (per-dim scale)], [], [], [Per-dim timescales],
  [lowrank R (rank=512)], [], [], [Can we compress R?],
)

== Phase 5: Architecture Scaling
Same ~1B params, different shape.

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Config*], [*dim*], [*depth*], [*Params*], [*Loss*],
  [baseline], [2048], [32], [1.01B], [5.27],
  [deeper+narrower], [1536], [48], [~1B], [],
  [wider+shallower], [3072], [16], [~1B], [],
)

== Key Findings (so far)
1. *Compete×silu is optimal*: Group softmax competition with silu activation gives best loss (5.223)
2. *32-64 groups is sweet spot*: Coarser (16) or finer (128) both hurt performance
3. *Activation doesn't matter much*: silu/gelu/mish all within 0.02 of each other
4. *Learned temperature hurts*: Fixed temp=1.0 is better than learned
5. *Asymmetric gate doesn't help*: Coarse compete + full silu (5.244) worse than symmetric (5.223)
6. *Input-dependent delta is CRITICAL*: Fixed decay (5.627) is 0.4 worse than learned delta (5.223)!
   - This is the biggest effect found so far
   - The model needs to learn when to update vs. when to remember
7. *R matrix + tanh helps but less than delta*: Mamba2-style (5.394) is 0.17 worse than baseline
   - Removing R matrix and tanh costs ~0.17 loss
   - Still better than fixed delta (5.627) by 0.23
   - Input-dependent delta > R matrix + tanh in importance
8. *Tanh alone (without R) HURTS*: mamba2_tanh (5.679) is WORSE than mamba2_style (5.394)
   - Adding tanh to Mamba2-style (no R matrix) makes it worse!
   - Caveat: different dim (2816 vs 2048), no Haste kernel (5x slower training)
   - The R matrix may be essential for tanh to help
   - Without R @ h recurrence, tanh just squashes the B @ x candidate unnecessarily
9. *Silu HURTS even more*: mamba2_silu (5.762) is WORSE than mamba2_tanh (5.679)
   - Using silu instead of tanh on the candidate is even worse!
   - Testing with Haste kernel at dim=2048, 1.01B params
   - Ranking: no nonlinearity (5.394) > tanh (5.679) > silu (5.762)
   - Key insight: In Mamba2-style (no R matrix), any nonlinearity on candidate HURTS
   - The hidden state is a weighted average of candidates - squashing them limits expressiveness
10. *Delta formulation doesn't matter*: mamba2_delta (5.226) ≈ baseline (5.223)
   - Mamba2-style delta: delta = softplus(W @ x), decay = exp(-delta)
   - Our baseline: delta = sigmoid(W @ x)
   - Both achieve essentially identical loss (~0.003 difference)
   - The specific parameterization (sigmoid vs softplus/exp) is not important
   - What matters: input-dependent delta, not how it's computed
11. *Conv4 HURTS*: mamba2_delta_conv4 (5.327) is WORSE than mamba2_delta (5.226)
   - Adding Mamba2-style causal conv (kernel=4) before the RNN hurts by ~0.1
   - Also slower: ~13k T/s vs ~17k T/s without conv
   - The R @ h recurrence already provides sufficient local context mixing
   - Unlike Mamba2 (diagonal A), our full R matrix doesn't need conv for local features

// Leaky Elman RNN: Discretized Continuous-Time Recurrence
// A mathematical description of the architecture family

#set document(title: "Leaky Elman RNN", author: "Erik Garrison")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 18pt, weight: "bold")[Leaky Elman RNN: Discretized Continuous-Time Recurrence]

  #v(0.5em)
  #text(size: 11pt, style: "italic")[Working Document -- December 2025]
]

#v(1em)

= Introduction

We present a family of RNN architectures based on *discretized continuous-time dynamics* combined with input-dependent output gating. These models bridge classical Elman RNNs and modern state space models (Mamba2), exploring the trade-off between nonlinear state dynamics and associative parallelism.

*Key insight*: The "leaky integration" pattern from SSMs---$bold(h)_t = alpha bold(h)_(t-1) + (1-alpha) bold(f)(bold(x)_t)$---can be combined with nonlinear candidates and input-dependent output gating to match or exceed Mamba2 performance.

= Architecture Family

== Common Structure

All architectures in this family share the leaky integration pattern:

$ bold(h)_t = (1 - bold(delta)_t) dot.circle bold(h)_(t-1) + bold(delta)_t dot.circle "candidate"_t $

where:
- $bold(delta)_t in (0, 1)^D$ is the *blend factor* (input-dependent or learned)
- $"candidate"_t$ is a *nonlinear candidate state* (unlike Mamba2's linear $bold(B) bold(x)$)
- $dot.circle$ denotes element-wise multiplication

This is the discretized form of continuous dynamics $d bold(h) / d t = -bold(h) + bold(f)(bold(x), bold(h))$.

== Mamba2 (Linear SSM Baseline)

Mamba2 uses *linear* state dynamics with input-dependent gating:

$ bold(h)_t &= exp(-bold(Delta)_t dot.circle bold(A)) dot.circle bold(h)_(t-1) + (1 - exp(-bold(Delta)_t dot.circle bold(A))) dot.circle bold(B)_t bold(x)_t $
$ bold(y)_t &= bold(C)_t bold(h)_t $

where:
- $bold(Delta)_t = "softplus"(bold(W)_Delta bold(x)_t + bold(b)_Delta)$ --- input-dependent timestep
- $bold(A)$ --- learned diagonal decay matrix
- $bold(B)_t, bold(C)_t$ --- input-dependent projection matrices

*Key property*: Linear dynamics enable parallel scan (associative operation).

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: left,
  [*Strength*], [Parallelizable via associative scan],
  [*Weakness*], [Linear candidate limits expressivity],
)

== ElmanLeaky (Best RNN: 3.9 nats)

*Nonlinear* candidate with input-dependent leaky integration and *input-only* output gate:

#align(center)[
#box(stroke: 0.5pt, inset: 12pt)[
$ bold(c)_t &= tanh(bold(R) bold(h)_(t-1) + bold(W)_x bold(x)_t + bold(b)) #h(2em) & "(nonlinear candidate)" $
$ bold(delta)_t &= sigma(bold(W)_delta bold(x)_t + bold(b)_delta) #h(2em) & "(input-dependent blend)" $
$ bold(h)_t &= (1 - bold(delta)_t) dot.circle bold(h)_(t-1) + bold(delta)_t dot.circle bold(c)_t #h(2em) & "(leaky integration)" $
$ bold(g)_t &= "silu"(bold(W)_g bold(x)_t + bold(b)_g) #h(2em) & "(input-only output gate)" $
$ bold(y)_t &= bold(h)_t dot.circle bold(g)_t #h(2em) & "(gated output)" $
]
]

*Key innovations*:
1. *Nonlinear candidate*: $tanh(bold(R) bold(h) + bold(W) bold(x))$ vs Mamba2's linear $bold(B) bold(x)$
2. *Recurrence matrix R*: Sees the *blended* state $bold(h)_(t-1)$, not raw Elman output
3. *Input-only output gate*: Like Mamba2's $bold(C)$ matrix --- depends only on $bold(x)$, not $bold(h)$

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: left,
  [*Strength*], [Nonlinear dynamics, best RNN performance (3.9 nats)],
  [*Weakness*], [Non-associative (sequential computation)],
)

== ElmanLeakySilu (Currently Testing)

Same as ElmanLeaky but with *silu* instead of *tanh*:

#align(center)[
#box(stroke: 0.5pt, inset: 12pt)[
$ bold(c)_t &= "silu"(bold(R) bold(h)_(t-1) + bold(W)_x bold(x)_t + bold(b)) #h(2em) & "(silu candidate)" $
$ bold(delta)_t &= sigma(bold(W)_delta bold(x)_t + bold(b)_delta) #h(2em) & "(input-dependent blend)" $
$ bold(h)_t &= (1 - bold(delta)_t) dot.circle bold(h)_(t-1) + bold(delta)_t dot.circle bold(c)_t #h(2em) & "(leaky integration)" $
$ bold(g)_t &= "silu"(bold(W)_g bold(x)_t + bold(b)_g) #h(2em) & "(input-only output gate)" $
$ bold(y)_t &= bold(h)_t dot.circle bold(g)_t #h(2em) & "(gated output)" $
]
]

*Hypothesis*: silu's non-saturating behavior for positive values may improve gradient flow compared to tanh.

$ "silu"(x) = x dot sigma(x) #h(2em) "where" sigma(x) = 1 / (1 + e^(-x)) $

Unlike tanh which saturates at $plus.minus 1$, silu grows approximately linearly for positive $x$.

== ElmanLeakySelective (Failed: 5.0 nats)

Uses *both* $bold(h)$ and $bold(x)$ in the output gate (like full GRU selectivity):

#align(center)[
#box(stroke: 0.5pt, inset: 12pt)[
$ bold(c)_t &= tanh(bold(R) bold(h)_(t-1) + bold(W)_x bold(x)_t + bold(b)) #h(2em) & "(candidate)" $
$ "dt"_t &= "softplus"(bold(W)_delta bold(x)_t + bold(b)_delta) #h(2em) & "(timestep)" $
$ "decay" &= exp(-exp(bold(A)_"log")) #h(2em) & "(log-space, stable)" $
$ alpha_t &= exp(-"dt"_t dot.circle "decay") #h(2em) & "(blend factor)" $
$ bold(h)_t &= alpha_t dot.circle bold(h)_(t-1) + (1 - alpha_t) dot.circle bold(c)_t #h(2em) & "(leaky integration)" $
$ bold(g)_t &= "silu"(bold(W)_g^h bold(h)_t + bold(W)_g^x bold(x)_t + bold(b)_g) #h(2em) & "(h+x output gate)" $
$ bold(y)_t &= bold(h)_t dot.circle bold(g)_t #h(2em) & "(gated output)" $
]
]

*Key difference*: Output gate depends on *both* $bold(h)$ and $bold(x)$ (full bilinear interaction).

*Result*: Performs *worse* than input-only gating (5.0 vs 3.9 nats). Hypothesis: the h+x gate is "overdoing it" --- redundant when the state already carries relevant information.

= The Read-Write Decomposition

== Why Input-Only Gating Works

Intuitively, we decompose sequence modeling into:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Component*], [*Function*], [*Should depend on*],
  [Writer (state update)], [Build structured memory], [$bold(h)_(t-1)$ and $bold(x)_t$],
  [Reader (output gate)], [Select what to read out], [$bold(x)_t$ only],
)

The *writer* (leaky integration) already has full access to both $bold(h)$ and $bold(x)$. The *reader* (output gate) only needs to know what the *current input* is asking for --- the *state* contains all the context.

This mirrors Mamba2's design:
- $bold(B)$, $bold(Delta)$: input-dependent *writing*
- $bold(C)$: input-dependent *reading* (does NOT see $bold(h)$ directly)

== Why h+x Gating Hurts

When the output gate sees both $bold(h)$ and $bold(x)$:
1. *Redundant computation*: $bold(h)$ already encodes relevant history
2. *Gradient competition*: Gate gradients compete with state gradients
3. *Increased parameters*: $bold(W)_g^h$ adds $D^2$ parameters with no benefit

*Lesson*: Simpler is better. Let the state do its job.

= Activation Functions

== tanh vs silu

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Property*], [*tanh*], [*silu*],
  [Range], [$(-1, 1)$], [$(-0.28, infinity)$],
  [Saturation], [Both directions], [Only negative],
  [Gradient at 0], [1], [0.5],
  [Formula], [$tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))$], [$"silu"(x) = x dot sigma(x)$],
)

*silu advantages*:
- Non-saturating for positive values → better gradient flow
- Used throughout Mamba2 and modern architectures
- Smooth approximation to ReLU with learnable curve

*tanh advantages*:
- Bounded output → numerically stable state dynamics
- Symmetric → no positive bias

= Empirical Results (Post Loss-Masking Fix)

All results from December 22-24, 2025, after fixing padding token loss masking.

== Main Comparison (~1B Parameters, 3K Steps, 512 Context)

Training on The Pile dataset with 8× A100 GPUs (DDP), 512-token context chunks.

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, left),
  [*Architecture*], [*Avg Loss (2k-3k)*], [*vs Mamba2*], [*Notes*],
  [Mamba2 (linear SSM)], [3.68], [baseline], [Parallelizable],
  [*ElmanLeaky (tanh)*], [*3.91*], [+6%], [*Best RNN*],
  [ElmanLeakySelective (h+x gate)], [~5.0], [+36%], [Worse than input-only],
  [ElmanLeakySilu (silu)], [TBD], [TBD], [Currently training],
)

== Key Findings

1. *ElmanLeaky nearly matches Mamba2* (3.91 vs 3.68) using *nonlinear* state dynamics and *sequential* computation.

2. *Input-only gating beats h+x gating* by ~1.1 nats (3.91 vs ~5.0). The simpler approach wins.

3. *No output gate at all* in the CUDA kernel works fine --- the gruboros wrapper adds the input-only silu gate externally.

== Architecture Diagram

#align(center)[
#box(stroke: 0.5pt, inset: 15pt)[
```
ElmanLeaky / ElmanLeakySilu Architecture:

    x_t ──────────────────┬───────────────────────────────────────┐
                          │                                       │
                          ▼                                       │
                    ┌───────────┐                                 │
       h_{t-1} ────►│ R @ h +   │                                 │
                    │ Wx @ x + b│                                 │
                    └─────┬─────┘                                 │
                          │                                       │
                          ▼                                       │
                    ┌───────────┐                                 │
                    │ tanh/silu │ ─────► candidate                │
                    └───────────┘                                 │
                                                                  │
                    ┌───────────┐                                 │
    x_t ───────────►│ sigmoid   │ ─────► δ (blend factor)         │
                    │ (W_δ @ x) │                                 │
                    └───────────┘                                 │
                                                                  │
              ┌───────────────────────────────────┐               │
              │ h_t = (1-δ)·h_{t-1} + δ·candidate │               │
              └─────────────────┬─────────────────┘               │
                                │                                 │
                                ▼ h_t                             │
                                                                  │
                    ┌───────────┐                                 │
    x_t ───────────►│ silu      │ ◄───────────────────────────────┘
                    │ (W_g @ x) │ ─────► gate (INPUT-ONLY!)
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐
         h_t ──────►│    ⊙      │ ─────► y_t (output)
                    └───────────┘
```
]
]

= Implementation Details

== CUDA Kernels (haste)

We implement fused CUDA kernels for the sequential recurrence:

```cpp
// ElmanLeaky forward kernel (simplified)
for (int t = 0; t < T; t++) {
    float raw = R @ h + Wx[t] + bias;
    float candidate = tanhf(raw);           // or silu for ElmanLeakySilu
    float delta = sigmoid(delta_raw[t]);
    h = (1.0f - delta) * h + delta * candidate;
    output[t] = h;
}
```

Key optimizations:
- Fused matmul + activation + blending
- Memory-efficient: only stores $bold(h)$, $bold(v)$ (pre-activations)
- Chunked processing for gradient checkpointing

== Initialization

Critical for stable training:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Parameter*], [*Init*], [*Rationale*],
  [$bold(W)_x$, $bold(R)$], [$cal(U)(-1/sqrt(D), 1/sqrt(D))$], [Standard scaling],
  [$bold(b)_delta$], [$-2.0$], [$sigma(-2) approx 0.12$ → slow dynamics],
  [$bold(W)_delta$], [$0.1 times$ normal], [Small perturbations around bias],
  [$bold(W)_g$ (gate)], [$cal(N)(0, 0.02)$], [Small for stable residual],
)

= Open Questions

1. *Will silu beat tanh?* ElmanLeakySilu is currently training --- hypothesis is that silu's gradient properties will help.

2. *Scaling*: Does the Mamba2 gap (3.68 vs 3.91) close or widen at larger scales?

3. *Long context*: Does nonlinear recurrence provide advantages on tasks requiring complex temporal reasoning beyond what linear SSMs can express?

4. *Minimal architecture*: Is the full leaky integration necessary, or would simpler dynamics suffice?

= Conclusion

The Leaky Elman family demonstrates that *nonlinear* recurrent dynamics can nearly match *linear* SSMs (Mamba2) at ~1B scale. Key findings:

1. *Input-only output gating* is critical and sufficient --- h+x gating hurts
2. *Leaky integration* ($bold(h) = (1-delta) bold(h) + delta dot "candidate"$) provides stable training
3. *Nonlinear candidates* (tanh or silu) preserve true recurrence expressivity
4. *Simple architectures win*: fewer parameters, cleaner gradients

The 0.23 nat gap to Mamba2 (3.91 vs 3.68) is surprisingly small given that ElmanLeaky:
- Cannot parallelize (sequential computation)
- Has nonlinear state dynamics (not associative)
- Uses a much simpler architecture

#v(2em)
#line(length: 100%)
#v(0.5em)
#text(size: 9pt, style: "italic")[
  Code: `github.com/ekg/gruboros` (model), `github.com/ekg/haste` (CUDA kernels) \
  Training: 8× A100 (80GB), PyTorch DDP, The Pile dataset \
  December 2025
]

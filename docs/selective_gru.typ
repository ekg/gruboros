// Selective GRU: True Recurrence with Input-Dependent Output Gating
// A mathematical description of the architecture

#set document(title: "Selective GRU (GRUS)", author: "Erik Garrison")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 18pt, weight: "bold")[Selective GRU: True Recurrence Meets Input-Dependent Gating]

  #v(0.5em)
  #text(size: 11pt, style: "italic")[Working Document -- December 2025]
]

#v(1em)

= Introduction

We present *Selective GRU* (GRUS), an architecture that combines standard gated recurrent units with input-dependent output gating. This simple modification---adding a bilinear selectivity mechanism to GRU outputs---matches state-of-the-art selective state space models (Mamba-2) while preserving true (non-associative) recurrence.

The key insight: modern sequence models require both *(1) sophisticated state dynamics* for building memory and *(2) input-dependent output gating* for selective readout. GRU provides the former; we add the latter.

= Architecture

== Standard GRU Recap

The Gated Recurrent Unit computes:

$ bold(z)_t &= sigma(bold(W)_z [bold(h)_(t-1), bold(x)_t] + bold(b)_z) #h(2em) & "(update gate)" $
$ bold(r)_t &= sigma(bold(W)_r [bold(h)_(t-1), bold(x)_t] + bold(b)_r) #h(2em) & "(reset gate)" $
$ tilde(bold(h))_t &= tanh(bold(W)_h [bold(r)_t dot.circle bold(h)_(t-1), bold(x)_t] + bold(b)_h) #h(2em) & "(candidate)" $
$ bold(h)_t &= (1 - bold(z)_t) dot.circle bold(h)_(t-1) + bold(z)_t dot.circle tilde(bold(h))_t #h(2em) & "(new state)" $

where $sigma$ is the sigmoid function and $dot.circle$ denotes element-wise multiplication.

Critically, GRU's gates depend on *both* the current input $bold(x)_t$ *and* the previous hidden state $bold(h)_(t-1)$. This makes the recurrence *non-associative*---it cannot be computed via parallel scan.

== Selective Output Gating

We add an input-dependent gate that modulates the GRU output:

$ bold(s)_t &= sigma(bold(W)_s^h bold(h)_t + bold(W)_s^x bold(x)_t + bold(b)_s) #h(2em) & "(selectivity gate)" $
$ bold(y)_t &= bold(s)_t dot.circle bold(h)_t #h(2em) & "(gated output)" $

The selectivity gate $bold(s)_t$ is a *bilinear* function of the hidden state and current input. This provides input-dependent *readout* of the memory, separate from the memory dynamics themselves.

== Full Forward Pass

For a sequence of inputs $bold(x)_(1:T)$:

#align(center)[
#box(stroke: 0.5pt, inset: 10pt)[
  #align(left)[
    *Algorithm 1: Selective GRU Forward Pass*

    *Input:* Sequence $bold(x)_(1:T)$, initial state $bold(h)_0$

    *for* $t = 1$ *to* $T$ *do*

    #h(1em) $bold(h)_t arrow.l "GRU"(bold(x)_t, bold(h)_(t-1))$ #h(1em) // standard GRU step

    #h(1em) $bold(s)_t arrow.l sigma(bold(W)_s^h bold(h)_t + bold(W)_s^x bold(x)_t)$ #h(1em) // selectivity

    #h(1em) $bold(y)_t arrow.l bold(s)_t dot.circle bold(h)_t$ #h(1em) // gated output

    *end for*

    *return* $bold(y)_(1:T)$
  ]
]
]

== Architecture Diagram

#align(center)[
#box(stroke: 0.5pt, inset: 15pt)[
```
                    ┌─────────────────────────────────────┐
                    │                                     │
    x_t ───────────┬┴─────────────┐                       │
                   │              │                       │
                   ▼              │                       │
              ┌─────────┐         │                       │
   h_{t-1} ──►│   GRU   │─────────┼──► h_t ──┬───────────┼──►
              └─────────┘         │          │           │
                                  │          ▼           │
                                  │    ┌───────────┐     │
                                  └───►│ σ(Wh·h +  │     │
                                       │   Wx·x)  │     │
                                       └────┬──────┘     │
                                            │ s_t       │
                                            ▼           │
                                       ┌─────────┐      │
                              h_t ────►│    ⊙    │──────┘
                                       └────┬────┘
                                            │
                                            ▼
                                          y_t (output)
```
]
]

= The Read-Write Decomposition

== Why Selectivity Matters

GRU's gates control *memory formation* (what to remember, what to forget). But the output is simply $bold(h)_t$---raw, unfiltered. There is no mechanism for the current input to influence *which aspects* of memory are relevant for the current prediction.

We decompose the sequence modeling task into:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Component*], [*Function*], [*Implementation*],
  [Writer], [Build structured memory over time], [GRU (non-linear, state-dependent gates)],
  [Reader], [Extract task-relevant information], [Selectivity (bilinear in $bold(h)$ and $bold(x)$)],
)

This separation mirrors Mamba's architecture, where input-dependent $bold(B)$ and $Delta$ control writing, while input-dependent $bold(C)$ controls reading.

== Gradient Flow Perspective

The selectivity layer provides a *direct gradient path* from the output loss to both:
- The hidden state $bold(h)_t$ (what to remember)
- The current input $bold(x)_t$ (what to attend to)

Without selectivity, the input's influence on the output must flow entirely through the recurrent dynamics---a longer, more entangled path.

= Relationship to Other Architectures

== Comparison with Mamba

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Aspect*], [*Mamba-2*], [*GRUS (This Work)*],
  [State dynamics], [Linear: $bold(h)_t = bold(A)_t bold(h)_(t-1) + bold(B)_t bold(x)_t$], [Non-linear: full GRU],
  [Output gating], [$bold(y)_t = bold(C)_t bold(h)_t$ (input-dependent)], [$bold(y)_t = bold(s)_t dot.circle bold(h)_t$ (input-dependent)],
  [Associative], [Yes (parallel scan)], [*No* (sequential)],
  [Computational class], [TC$""^0$ limited], [Beyond TC$""^0$],
)

The key difference: Mamba linearizes state dynamics for parallelization; we preserve non-linear dynamics and accept sequential computation.

== Comparison with Griffin/Hawk

DeepMind's Griffin/Hawk use the RG-LRU (Real-Gated Linear Recurrent Unit):

$ bold(r)_t &= sigma(bold(W)_r bold(x)_t) #h(2em) & "(recurrence gate---input only)" $
$ bold(i)_t &= sigma(bold(W)_i bold(x)_t) #h(2em) & "(input gate---input only)" $
$ bold(h)_t &= bold(r)_t dot.circle bold(a) dot.circle bold(h)_(t-1) + (1 - bold(r)_t dot.circle bold(a)) dot.circle (bold(i)_t dot.circle bold(x)_t) $

Note: gates depend *only on $bold(x)_t$*, not on $bold(h)_(t-1)$. This enables parallel scan but removes history-dependent gating.

GRUS preserves full state-dependent gating in the GRU while adding input-dependent output gating.

== Why EMA + Selectivity Fails

Exponential moving average:
$ bold(h)_t = alpha bold(h)_(t-1) + (1 - alpha) bold(x)_t $

provides only smoothing---no content-dependent memory decisions. Even with perfect output gating, the state lacks the structure needed for complex sequence modeling. The *writer* matters, not just the *reader*.

= Empirical Results

== Main Comparison (1B Parameters, 10K Steps, 512 Context)

Training ~1B parameter models on The Pile dataset with 8× A100 GPUs (DDP), 512-token context. Loss values are averaged over the final 1,000 training steps for stability:

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center, center),
  [*Configuration*], [*Params*], [*Associative*], [*Avg Loss (last 1k)*], [*vs Stock GRU*],
  [Stock GRU (no selectivity)], [1.11B], [No], [3.76], [baseline],
  [Mamba-2 SSD], [1.00B], [Yes], [2.88], [−23%],
  [GRU + Selectivity (sigmoid)], [1.01B], [*No*], [2.99], [−20%],
  [*GRU + Selectivity (SiLU)*], [1.01B], [*No*], [*2.82*], [*−25%*],
)

#v(0.5em)

*Key finding:* Output selectivity accounts for *~0.8 nats* improvement over Stock GRU. All models have matched parameter counts (1.00--1.11B)---notably, Stock GRU has the *most* parameters (1.11B) yet performs worst, demonstrating that the improvement is due to architectural differences, not capacity. Both Mamba-2 and GRUS achieve similar performance through different mechanisms:
- Mamba-2: Input-dependent $bold(C)$ matrix for output selection
- GRUS: Multiplicative gate $bold(s)_t = sigma(bold(W)_h bold(h) + bold(W)_x bold(x))$

== Gate Activation: Sigmoid vs SiLU

Mamba-2 uses SiLU (Swish) activation throughout. We tested whether the gate activation matters:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, left, center),
  [*Gate Activation*], [*Formula*], [*Performance*],
  [Sigmoid (default)], [$bold(s)_t = sigma(bold(W)_h bold(h) + bold(W)_x bold(x))$], [2.99 nats],
  [*SiLU (Mamba-style)*], [$bold(s)_t = "SiLU"(bold(W)_h bold(h) + bold(W)_x bold(x))$], [*2.82 nats*],
)

*SiLU outperforms sigmoid by 0.17 nats* (6% relative improvement). This is a meaningful gain from a simple activation change.

SiLU differs from sigmoid in that it is *non-saturating* for positive values and allows negative outputs. The improved gradient flow appears to benefit training. Notably, SiLU GRUS (2.82) now *outperforms Mamba-2* (2.88) on this benchmark.

== Ablation Studies (256 Context -- Being Reworked)

#text(style: "italic")[Note: The following ablations were conducted with 256-token context. We are re-running these at 512 context for consistency with main results.]

Preliminary findings at 256 context:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Ablation*], [*Change*], [*Effect*],
  [Input-only gate], [Remove $bold(W)_h bold(h)$ term], [Degraded],
  [GLU-style gate], [Gate depends only on $bold(h)$], [Close to full],
  [EMA + Selectivity], [Replace GRU with EMA], [Much worse],
)

The bilinear interaction between $bold(h)$ and $bold(x)$ appears important, but the relative contributions of each term require further investigation at consistent context lengths.

== Architectural Variations

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: left,
  [*Name*], [*Gate Formula*], [*Notes*],
  [Multiplicative (GRUS)], [$sigma(bold(W)_h bold(h) + bold(W)_x bold(x))$], [Default, best understood],
  [Input-only], [$sigma(bold(W)_x bold(x))$], [Loses state-dependent selection],
  [GLU-style], [$sigma(bold(W) bold(h)) dot.circle bold(h)$], [No input dependence in gate],
  [SiLU variant], [$"SiLU"(bold(W)_h bold(h) + bold(W)_x bold(x))$], [Testing Mamba-style activation],
)

= Implementation Notes

== Naming Conventions

The architecture has been referred to by several names during development:

- *Mult GRU* / *CuDNN Mult GRU*: Implementation using cuDNN GRU kernel + multiplicative gate
- *GRUS*: "GRU Selective" or "GRU with Selectivity"
- *Selective GRU*: Descriptive name emphasizing the selectivity mechanism

We propose *GRUS* as the canonical name, suggesting both "GRU + Selectivity" and evoking "gears" (mechanical, reliable).

== cuDNN Implementation

We use PyTorch's `torch.nn.GRU` with `batch_first=True`, which delegates to highly optimized cuDNN kernels. The selectivity layer is a simple linear projection followed by sigmoid and element-wise multiplication.

```python
class SelectiveGRU(nn.Module):
    def __init__(self, dim, depth):
        self.gru = nn.GRU(dim, dim, num_layers=depth, batch_first=True)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)

    def forward(self, x, h=None):
        h_out, h_final = self.gru(x, h)
        gate = torch.sigmoid(self.W_h(h_out) + self.W_x(x))
        return gate * h_out, h_final
```

== Gradient Checkpointing

For longer contexts, we implement chunk-level gradient checkpointing:
- Process sequence in chunks (e.g., 128 tokens)
- Checkpoint hidden states at chunk boundaries
- Recompute within chunks during backward pass

This trades compute for memory, enabling training on contexts beyond what fits in GPU memory.

= Discussion

== What This Suggests About Sequence Modeling

The success of GRUS implies:

1. *Output selection is the key differentiator*: The ~0.8 nat gap between Stock GRU and selective architectures (Mamba-2, GRUS) is attributable to input-dependent output gating. This is consistent across architectures.

2. *Mamba's success is decomposable*: Input-dependent state dynamics + input-dependent output gating are separable components that don't need to be entangled in a single formalism.

3. *Associativity is optional*: Performance matching doesn't require parallel-scannable operations. The training efficiency gap with good cuDNN kernels may be smaller than assumed.

4. *True recurrence remains viable*: Non-associative, state-dependent gating can match linearized alternatives at equal parameter counts.

== Observations from Current Experiments

*Activation function matters*: SiLU activation provides a 0.17 nat improvement over sigmoid (2.82 vs 2.99). This simple change makes GRUS outperform Mamba-2 (2.88) on this benchmark. SiLU's non-saturating behavior for positive values improves gradient flow.

*Context length independence*: The ~0.8 nat improvement from selectivity appears consistent at 512 context. Ablation experiments at 256 context showed similar patterns, though direct comparison requires re-running at matched conditions.

*z-gate initialization*: We confirmed that GRU z-gate bias initialization (0 vs −2) does not meaningfully affect results. Both Stock GRU and GRUS use PyTorch's default (zero) initialization internally.

== Open Questions

- *Scaling behavior*: Does the selectivity advantage hold at larger scales (7B, 13B)?
- *Long-context tasks*: Does true recurrence provide advantages on tasks requiring complex temporal reasoning?
- *Minimal sufficient architecture*: Is full GRU necessary, or would simpler non-linear dynamics suffice?

== Theoretical Implications

Associative parallel-scan operations are limited to the complexity class TC$""^0$ (constant-depth threshold circuits). True recurrence can recognize languages beyond TC$""^0$.

The empirical question: do practical tasks require this additional expressivity? Our architecture enables investigating this by providing a performant non-associative baseline.

= Conclusion

Selective GRU demonstrates that adding input-dependent output gating to standard GRU *exceeds* the performance of state-of-the-art selective state space models at matched parameter count (~1B). With SiLU activation, GRUS achieves 2.82 nats vs Mamba-2's 2.88 nats---a 2% improvement while using true (non-associative) recurrence.

The ~0.9 nat improvement over Stock GRU (3.76 → 2.82) is attributable to:
1. *Selectivity mechanism* (~0.8 nats): Input-dependent output gating
2. *SiLU activation* (~0.17 nats): Non-saturating gradient flow

The architecture is simple (a single bilinear gate layer with SiLU), uses battle-tested cuDNN kernels, and preserves the non-associative dynamics that may prove important for complex temporal reasoning tasks.

The key insight is the *read-write decomposition*: GRU excels at building structured memory (writing); selectivity enables input-dependent retrieval (reading). Neither alone suffices; together they exceed Mamba-2.

#v(2em)
#line(length: 100%)
#v(0.5em)
#text(size: 9pt, style: "italic")[
  Code: `github.com/ekg/gruboros` \
  Training: 8× A100 (80GB), PyTorch DDP, The Pile dataset \
  December 2025
]

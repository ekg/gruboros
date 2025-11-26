# Training Investigation Report V2: HybridFusedGRU Analysis

**Date**: 2025-11-26
**Correction**: You're training **HybridFusedGRU (StandardGRU)**, NOT minGRU!

---

## Critical Discovery: Log-Space vs Normal-Space Gating

### The Smoking Gun

**HybridFusedGRU** (`mingru/hybrid_fused_gru.py` lines 115-123):
```python
# Normal-space operations
r = tl.sigmoid(i_r + h_r)
z = tl.sigmoid(i_z + h_z)
n_pre = i_n + r * h_n
n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
exp_2x = tl.exp(2.0 * n_pre_clamped)
n = (exp_2x - 1.0) / (exp_2x + 1.0)
h = (1.0 - z) * h + z * n  # ← NORMAL SPACE interpolation
```

**minGRU** (`mingru/minGRU.py` lines 101-114):
```python
# Log-space operations
log_coeffs = -F.softplus(gate)  # log(1 - sigmoid(gate))
log_z = -F.softplus(-gate)      # log(sigmoid(gate))
log_tilde_h = log_g(hidden)
log_values = log_z + log_tilde_h
# ...
log_out = heinsen_associative_scan_log(log_coeffs, log_values)
# ← Stays in LOG SPACE until final output projection
```

### Why This Matters

**Normal-space interpolation** (HybridFusedGRU):
```
h_new = (1 - z) * h_old + z * n
```

When backpropagating through 20 layers, gradient becomes:
```
∂L/∂h_0 = ∂L/∂h_20 × Π(1 - z_i) for i=1..20
```

**Vanishing gradient calculation:**
- If each (1 - z_i) ≈ 0.95: 0.95^20 ≈ **0.36** (64% gradient loss)
- If each (1 - z_i) ≈ 0.90: 0.90^20 ≈ **0.12** (88% gradient loss)
- If each (1 - z_i) ≈ 0.85: 0.85^20 ≈ **0.04** (96% gradient loss!)

**Log-space operations** (minGRU):
- All multiplications become additions in log space
- Prevents underflow/overflow
- More stable gradient propagation
- Comment in code: "This was the main source of numerical instability - now fixed"

---

## How Mamba Handles Deep Networks

**Sources:**
- [DataCamp - Mamba Architecture](https://www.datacamp.com/tutorial/introduction-to-the-mamba-llm-architecture)
- [Medium - Mamba Leap Forward](https://medium.com/@puneetthegde22/mamba-architecture-a-leap-forward-in-sequence-modeling-370dfcbfe44a)
- [ArXiv - Mamba Paper](https://arxiv.org/pdf/2312.00752)

### Key Techniques Mamba Uses (that we DON'T):

1. **Residual Connections**
   > "The Mamba architecture repeats blocks interleaved with standard normalization and residual connections"

   > "Before entering the Mamba block, a copy of the input is sent directly to the end as a residual connection"

2. **Layer Normalization**
   > "Dropout and layer normalization are utilized after each Mamba block to enhance model robustness, prevent overfitting, and accelerate training convergence"

3. **Log-Space Stability**
   > "In Mamba's implementation, log(A) is tracked for numerical stability during training"

   > "Selective SSMs maintain stability by ensuring eigenvalues of the discrete recurrences have absolute value bounded by 1"

4. **Default Depth: 32 layers** in HuggingFace implementation
   - Uses ALL the above techniques to enable this depth

### Our Model Has NONE Of These:
- ❌ No residual connections between GRU layers
- ❌ No layer normalization between layers
- ❌ No log-space gating (uses normal-space)
- ✓ Has depth 20 (aggressive without the above)

---

## Research: Standard GRU vs Minimal GRU

**Sources:**
- [Minimal Gated Unit Paper](https://arxiv.org/abs/1603.09420)
- [Stack Overflow - How GRU Solves Vanishing Gradients](https://stats.stackexchange.com/questions/556070/how-gru-solves-vanishing-gradient)
- [GeeksforGeeks - GRU Networks](https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/)

### Standard GRU Issues at Depth:

**Gradient Explosions:**
> "GRUs suffer from gradient explosions due to their nonlinear dynamics. The dynamics can drastically change when the parameters cross certain values, called bifurcation points, in the learning process. Therefore, the gradient of the state with respect to the parameters can drastically increase at a bifurcation point."

**Gating Limitations:**
> "GRUs help mitigate [vanishing gradients] by using gates that regulate the flow of gradients during training, ensuring that important information is preserved and that gradients do not shrink excessively over time."

BUT this only works for **shallow** networks (2-6 layers). At depth 20, the multiplicative effect compounds.

### Minimal GRU (MGU) Advantages:

> "MGU only contains one gate, which is a minimal design among all gated hidden units. The update and reset gate vector is merged into a forget gate."

> "Experiments on various sequence data show that MGU has comparable accuracy with GRU, but has a simpler structure, fewer parameters, and faster training."

**Key insight**: Simpler gating = more stable training, especially in log-space formulation.

---

## Parameter Count Analysis

### Current Configuration (20 layers, depth=20, dim=2048, expansion=1.0):

Per-layer parameters (expansion=1.0, dim_inner=dim):
- Input projection: dim × 3×dim = 2048 × 6144 = 12.6M
- Hidden projection: dim × 3×dim = 2048 × 6144 = 12.6M
- Output projection: none (Identity for expansion=1.0)
- **Total per layer: ~25M params**

For 20 layers: 20 × 25M = **500M params** (plus embedding/output head = ~700M total)

### To Match 700M Params with 3 Layers:

Target: 700M / 3 layers = **233M params per layer**

**Option 1: expansion=1.0**
- Per layer: 2 × (dim × 3×dim) = 6×dim²
- 233M = 6×dim²
- dim = sqrt(38.8M) = **6,230**
- ✓ No expansion complexity
- ✗ **Absolutely massive** matrix multiplications (6230×6230)

**Option 2: expansion=2.0** (matching minGRU paper)
- Per layer: dim_inner = 2×dim
- Per layer: 2 × (dim × 3×2×dim) = 12×dim²
- 233M = 12×dim²
- dim = sqrt(19.4M) = **4,405**
- ✓ More manageable than 6230
- ✓ Matches paper's expansion=2.0
- ✗ Still very large matmuls (4405×8810)

**Option 3: Go wider AND add dropout**
- Use 3 layers, dim=4096, expansion=2.0, **dropout=0.2**
- Params: 3 × 12×(4096²) ≈ 600M (close enough)
- Add dropout between layers for regularization
- Dropout helps gradient flow (per research)

---

## Three Potential Fixes

### Fix 1: Implement Log-Space Gating in HybridFusedGRU ✅ (Best for depth)

**Convert HybridFusedGRU to log-space** like minGRU:

```python
# Current (normal space):
h = (1.0 - z) * h + z * n

# Log-space version:
log_h = torch.logaddexp(
    log_one_minus_z + log_h_prev,
    log_z + log_n
)
```

**Pros:**
- Maintains depth=20 capability
- Fixes numerical stability root cause
- Proven to work (minGRU paper)
- Keeps current architecture investment

**Cons:**
- Requires modifying Triton kernels (complex)
- Need to carefully handle log-space throughout
- May need extensive testing/debugging

### Fix 2: Add Residual Connections + LayerNorm ✅ (Mamba approach)

**Add between-layer connections:**

```python
# In model.py, between GRU layers:
for i, layer in enumerate(self.layers):
    residual = x
    x = layer_norm(x)
    x = gru_layer(x, prev_hiddens[i])
    x = x + residual  # Skip connection
```

**Pros:**
- Industry-standard solution (Mamba, Transformers)
- Enables arbitrary depth
- Relatively simple to implement
- Doesn't require kernel changes

**Cons:**
- Changes model architecture significantly
- May not fully solve HybridFusedGRU's normal-space instability
- Need to validate it doesn't break training

### Fix 3: Go Shallow and Wide ✅ (Immediate, proven)

**3 layers, dim=4096-6230, expansion=2.0, dropout=0.2**

**Pros:**
- Matches minGRU paper's successful approach
- Avoids depth-related vanishing gradients entirely
- Can start training immediately
- Proven to work (minGRU achieved loss 1.548 with 3 layers)

**Cons:**
- **Massive matrix multiplications** (4096×8192 or 6230×6230)
- May be slower per-step than deep network
- Less parameter efficiency than deep+narrow

---

## Root Cause: Why Gradients Die

### Mathematical Explanation:

In StandardGRU (HybridFusedGRU), the hidden state update is:
```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t
```

Gradient backpropagation through time/depth:
```
∂h_t/∂h_{t-1} = (1 - z_t)
```

For a 20-layer network, gradient becomes:
```
∂L/∂h_0 = ∂L/∂h_20 × ∏_{i=1}^{20} (1 - z_i)
```

**This is a product of 20 terms, each < 1!**

If the update gates z are functioning normally (z ≈ 0.1 to 0.5):
- (1 - z) ranges from 0.5 to 0.9
- Product of 20 such terms: 0.7^20 ≈ 0.0008 (**99.92% gradient loss!**)

### Why Our Experiments All Failed:

| Experiment | Result | Why It Failed |
|------------|---------|---------------|
| Increase LR 2× → 100× | Gradients still died | Multiplicative decay through 20 layers overwhelms any LR |
| Fresh optimizer | Temporary spike, then death | Doesn't fix architecture's gradient path |
| SGD (no momentum) | Same death pattern | Optimizer can't fix broken gradient flow |
| Remove grad clipping | Helped initially, then died | Clipping wasn't root cause, just symptom |
| Schedule-Free AdamW | Same as regular AdamW | Adaptive LR can't overcome vanishing gradients |

**All optimizers face the same problem**: Gradients multiply through 20 layers of (1-z) < 1.

### Why Log-Space Fixes This:

In log-space, **multiplication becomes addition**:
```
log(a × b × c) = log(a) + log(b) + log(c)
```

So instead of:
```
gradient = g_20 × (1-z_19) × (1-z_18) × ... × (1-z_1)
          ≈ g_20 × 0.0008  [VANISHES!]
```

We get (in log-space):
```
log_gradient = log(g_20) + log(1-z_19) + ... + log(1-z_1)
             = log(g_20) + sum of logs  [STABLE!]
```

Addition in log-space is numerically stable - no underflow!

---

## Why Depth=20 Seemed Reasonable (But Wasn't)

### Valid Reasoning:
1. ✓ Mamba uses 24-32 layers successfully
2. ✓ Deep Transformers use 24-96 layers
3. ✓ More layers = more expressiveness

### What We Missed:
1. ❌ Mamba uses **residual connections** at every layer
2. ❌ Mamba uses **layer normalization** between blocks
3. ❌ Mamba uses **log-space** for numerical stability
4. ❌ StandardGRU (HybridFusedGRU) has **none of these**!

**Analogy**: Trying to build a 20-story building without:
- Steel reinforcement (residual connections)
- Foundation stabilization (layer norm)
- Proper materials (log-space numerics)

The building might stand, but it won't be stable or functional.

---

## Recommended Path Forward

### Immediate Action (High Confidence):

**Test Fix 3 first**: 3 layers, go wide, match minGRU paper
```bash
--depth 3
--dim 4096  # or calculate for exact 700M
--expansion_factor 2.0  # match paper
--dropout 0.2  # match paper
--lr 0.001  # already correct
--grad_clip 0.0  # we found clipping harmful
--schedulefree  # keep using Schedule-Free AdamW
```

**Why this first**:
- Proven to work (minGRU paper: loss 1.548 with 3 layers)
- Eliminates depth-related vanishing gradients
- Can start training immediately (no code changes)
- Validates that our training pipeline works correctly

### Medium Term (If shallow works):

**Implement Fix 1 or Fix 2** to enable depth:

**Option A: Log-space HybridFusedGRU**
- Port log-space gating from minGRU to HybridFusedGRU
- Modify Triton kernels to work in log-space
- Test on 3-layer model first, then scale to 20

**Option B: Add architectural improvements**
- Residual connections between layers
- Layer normalization between layers
- Gradient checkpointing for memory efficiency
- Then test scaling to 10, 15, 20 layers

### Long Term (Research):

**Compare architectures systematically**:
1. Pure minGRU (log-space, 3 layers) - baseline
2. HybridFusedGRU shallow (3 layers, wide)
3. HybridFusedGRU with log-space (20 layers)
4. HybridFusedGRU with residuals+LN (20 layers)

Measure: loss, perplexity, training speed, memory usage.

---

## Questions To Investigate

### 1. Hidden State Continuity:
**Status**: Likely NOT the primary issue (log-space/depth is)

Evidence from code review:
- Hidden states properly detached (train.py:2150)
- Document boundaries reset correctly (train.py:2078-2084)
- No obvious accumulation bugs

BUT: Worth testing with `--reset_hidden_every_batch` to rule out completely.

### 2. Is HybridFusedGRU Implementation Correct?

**Potential Issues Found:**

**Line 123** (`hybrid_fused_gru.py`):
```python
h = (1.0 - z) * h + z * n  # Update h for next iteration!
```

This looks correct for StandardGRU, BUT:
- No log-space protection against underflow
- No numerical stability safeguards besides clamping n_pre to [-3, 3]
- When z ≈ 0 or z ≈ 1, this can be numerically unstable

**Line 199-200** (same pattern in regular cell):
```python
h_new = (1.0 - z) * h_prev + z * n
```

Same issue - normal-space interpolation.

**Compare to minGRU's approach** (lines 84-91):
```python
log_out = torch.logaddexp(
    log_one_minus_gate + prev_hidden,  # log(prev * (1-g))
    log_gate + log_hidden             # log(hidden * g)
)
```

Uses `logaddexp` which is numerically stable for log-space addition.

### 3. Do We Need Massive Matmuls?

**For 3 layers at 700M params: YES**

Calculation confirmed above:
- dim=6230 for expansion=1.0
- dim=4405 for expansion=2.0

**Are these practical?**
- Modern GPUs handle 4k-8k matmuls efficiently (optimized for Transformers)
- cuBLAS/Triton are highly optimized for large matmuls
- May be slower per-step but fewer steps to converge (minGRU: 575 vs our 337,000+)

**Memory concern:**
- Current: 20 layers × 2048 hidden = 40 × 2048 = ~80K hidden state elements
- Shallow: 3 layers × 4096 hidden = 12 × 4096 = ~50K hidden state elements
- **Actually uses LESS memory for hidden states!**

---

## Target Loss/Perplexity

### Byte-Level Language Modeling:

**Sources:**
- [Character-Level LM Paper](https://arxiv.org/pdf/1808.04444)
- [Tokenizer-Free Models](https://arxiv.org/pdf/1908.10322)

> "Byte-level models report results in bits per byte (bpb). State-of-art byte models achieve 0.874 bpb on the One Billion Word benchmark."

**Converting our loss to bpb:**
- Our loss: 4.7-4.8 (cross-entropy, nats)
- Convert to bits: 4.75 / ln(2) = 4.75 / 0.693 = **6.85 bpb**
- State-of-art: **0.874 bpb**
- We're **7.8× worse** than state-of-art!

**MinGRU on Shakespeare:**
- Loss: 1.548 nats
- In bpb: 1.548 / 0.693 = **2.23 bpb**
- We're **3× worse** than minGRU's 3-layer model!

**Realistic target for our 700M model:**
- Short term: Get below 3.0 loss (4.33 bpb)
- Medium term: Get below 2.0 loss (2.89 bpb)
- Long term: Approach 1.0-1.5 loss (1.44-2.16 bpb)

---

## Conclusion

### Root Causes (Revised Understanding):

1. **Primary**: HybridFusedGRU uses **normal-space gating** instead of log-space
   - Causes numerical instability at depth
   - Gradients multiply by (1-z) through each layer
   - 20 layers: 0.7^20 = 0.0008 (99.92% gradient loss!)

2. **Secondary**: No architectural supports for depth
   - No residual connections
   - No layer normalization
   - No dropout

3. **Tertiary**: Gradient clipping made it worse
   - But removing it didn't solve root cause

### The Path Forward:

**Immediate** (this week):
- ✅ Train 3-layer wide model (dim=4096, expansion=2.0, dropout=0.2)
- ✅ Use Schedule-Free AdamW, lr=0.001, no grad clipping
- ✅ Validate we can achieve loss < 2.0 like minGRU paper

**Medium term** (next week):
- ⚠️ Implement log-space gating in HybridFusedGRU
- OR implement residual connections + layer normalization
- Test scaling from 3 → 6 → 10 → 20 layers

**Long term** (research):
- Compare minGRU vs HybridFusedGRU performance
- Determine optimal depth/width tradeoff
- Benchmark training speed vs loss improvement

---

## Sources

### Mamba and SSM Architecture:
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/pdf/2312.00752)
- [DataCamp - Mamba Architecture](https://www.datacamp.com/tutorial/introduction-to-the-mamba-llm-architecture)
- [Theoretical Foundations of Deep SSMs](https://arxiv.org/html/2402.19047v1)
- [Towards Data Science - Mamba](https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451/)

### GRU Training and Stability:
- [Minimal Gated Unit Paper](https://arxiv.org/abs/1603.09420)
- [Stack Overflow - How GRU Solves Vanishing Gradients](https://stats.stackexchange.com/questions/556070/how-gru-solves-vanishing-gradient)
- [GeeksforGeeks - GRU Networks](https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/)
- [Preventing Gradient Explosions in GRUs](https://dl.acm.org/doi/pdf/10.5555/3294771.3294813)

### Vanishing Gradients:
- [Vanishing Gradient - Wikipedia](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- [GeeksforGeeks - Gradient Problems](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)

### Language Modeling:
- [Character-Level Language Modeling](https://arxiv.org/pdf/1808.04444)
- [Tokenizer-Free Language Models](https://arxiv.org/pdf/1908.10322)
- [HuggingFace - Perplexity](https://huggingface.co/docs/transformers/en/perplexity)

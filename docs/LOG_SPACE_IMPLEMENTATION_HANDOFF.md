# Log-Space RNN Implementation Handoff

**Date**: December 27, 2025
**Context**: Mamba2 beats Triple R by 7% at 1000 steps. Investigation suggests log-space computation as a potential key factor.
**Goal**: Implement log-space Elman/Triple R dynamics to test whether this matches Mamba2's numerical stability at depth.

---

## Executive Summary

Mamba2 uses a specialized **log-space segment-sum algorithm** for numerical stability during deep training. Without this, "the basic SSD algorithm produces NaNs immediately during training (even with FP32)" (Tri Dao). Our Triple R architecture lacks this, which **may** explain:

1. Mamba2's widening lead over training (2.7% → 7%)
2. Gradient death after ~350k steps in our models
3. Why adding Mamba2's selective gates to Triple R had zero effect

**Proposed fix**: Implement log-space dynamics for our Elman variants.

---

## ⚠️ Important Caveats

### This Is a Hypothesis, Not a Proven Solution

1. **Diagonal MHTR experiment is still running** - We don't yet know if diagonal structure alone helps
2. **Other potential causes not ruled out**:
   - Learning rate schedule
   - Layer norm placement (pre vs post)
   - Initialization schemes
   - Residual connection scaling
3. **Mamba2's advantage might be elsewhere**: SSD parallelization, multi-head structure, training dynamics

### Two Distinct Problems

| Problem | Location | Cause | Fix |
|---------|----------|-------|-----|
| Within-sequence underflow | Single layer, T tokens | Cumulative decay products | Log-space scan |
| Cross-layer gradient death | Deep network, L layers | RNN output → 0, gradients vanish | Unclear (residuals should help) |

Mamba2's log-space addresses within-sequence underflow. Our gradient death at 350k steps might be a different problem.

### Implementation Options

This document presents **two approaches**:

| Option | Recurrence | Complexity | Expressivity |
|--------|------------|------------|--------------|
| **A: Diagonal R** (Mamba2-style) | `h * r` element-wise | Low | Lower (no cross-dim interaction) |
| **B: Full R with logsumexp** | `R @ h` via logsumexp | High (~2x overhead) | Full (preserves current arch) |

Choose based on what experiments reveal.

---

## Quick Reference: The Ablation Ladder

```
STOCK ELMAN                    MAMBA2 TARGET: 3.924 avg50
    │
    ▼ +input gate (delta)
GATED ELMAN
    │
    ▼ +output selectivity (compete×silu)
SELECTIVE ELMAN  ◄─── We know this helps
    │
    ├──────────────────────────────────┐
    ▼ diagonal W_h                     ▼ keep full W_h
DIAGONAL SELECTIVE              FULL SELECTIVE
    │                                  │
    ▼ +log storage                     ▼ +logsumexp matmul
LOG-STORAGE DIAGONAL            LOG-COMPUTE FULL
    │                                  │
    └──────────────┬───────────────────┘
                   ▼ +R_delta modulation
               TRIPLE R
```

**Goal**: Find the simplest level that matches Mamba2 (3.924 avg50).

See **Part 9** for full experiment plan.

---

## Part 1: The Numerical Problem

### 1.1 Why Deep Recurrence Fails

In any recurrent model, hidden state evolves multiplicatively:

```python
h_t = A_t · h_{t-1} + B_t · x_t
```

For a sequence of length T with L layers, the effective computation involves:

```
Total multiplicative steps = L × T = 21 layers × 512 tokens = 10,752 steps
```

Even with decay factors close to 1:
- A = 0.95: 0.95^10752 ≈ 10^-240 (underflow)
- A = 0.99: 0.99^10752 ≈ 10^-47 (still underflow in fp16/bf16)

### 1.2 The Gradient Problem

Backpropagation through this chain requires:

```
∂L/∂h_0 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_1/∂h_0
        = ∂L/∂h_T · A_T · A_{T-1} · ... · A_1
```

Same product, same underflow → **vanishing gradients**.

### 1.3 Full R Matrices Are Worse

With diagonal A (Mamba2):
```python
h_t = a_t * h_{t-1}  # Element-wise, each dimension independent
```

With full R (Triple R):
```python
h_t = R @ h_{t-1}  # Matrix multiplication, dimensions interact
```

Full matrices have eigenvalues that can:
- Amplify some directions (exploding)
- Suppress others (vanishing)
- Create oscillatory dynamics (training instability)

The spectral radius of R^n grows/shrinks exponentially with n.

---

## Part 2: Mamba2's Solution

### 2.1 Log-Space Cumulative Products

Instead of computing products directly:
```python
# Linear space (underflows)
prod = a_1 * a_2 * a_3 * ... * a_T
```

Convert to log-space:
```python
# Log space (additions, no underflow)
log_prod = log(a_1) + log(a_2) + log(a_3) + ... + log(a_T)
prod = exp(log_prod)
```

### 2.2 The Segment-Sum Problem

For SSM computation, you need **all prefix products**:
```
P_1 = a_1
P_2 = a_1 * a_2
P_3 = a_1 * a_2 * a_3
...
```

Naive log-space approach:
```python
log_cumsum = cumsum([log(a_1), log(a_2), ...])  # [s_1, s_2, s_3, ...]
# To get product from i to j: exp(s_j - s_i)
```

**Problem**: When s_j and s_i are both large (e.g., 1000.0 and 999.5), subtracting causes **catastrophic cancellation** - you lose precision in the difference.

### 2.3 Segment-Sum (segsum) Algorithm

Mamba2's solution computes segment products **without subtraction**:

```python
def segsum(x):
    """Compute segment cumulative sums without subtraction.

    For input [a, b, c, d], produces matrix:
    [[a,   0,     0,     0    ],
     [a+b, b,     0,     0    ],
     [a+b+c, b+c, c,     0    ],
     [a+b+c+d, b+c+d, c+d, d  ]]

    Each row i contains cumsum starting from position j for j <= i.
    No subtraction needed - direct computation.
    """
    T = len(x)
    # Create lower triangular mask
    mask = torch.tril(torch.ones(T, T))
    # Broadcast and sum: result[i,j] = sum(x[j:i+1])
    x_expanded = x.unsqueeze(0).expand(T, T)
    result = (x_expanded * mask).cumsum(dim=1)
    return result
```

The key insight: Instead of computing one cumsum and subtracting, compute **T independent cumsums** starting from each position. This avoids catastrophic cancellation entirely.

### 2.4 Full SSD Log-Space Forward Pass

From Mamba2's minimal implementation:

```python
def ssd_minimal(X, A, B, C, block_len=64):
    """
    X: (batch, length, d_model) - input
    A: (batch, length, nheads) - diagonal state decay (log-space!)
    B: (batch, length, nheads, d_state) - input projection
    C: (batch, length, nheads, d_state) - output projection

    Returns: (batch, length, d_model)
    """
    # A is already in log-space: A = -softplus(A_raw)
    # This ensures A < 0 (decay), and we work with log(exp(A)) = A

    # Compute cumulative decay in log-space
    A_cumsum = A.cumsum(dim=1)  # log(a_1), log(a_1*a_2), ...

    # For each position i, compute decay from j to i
    # Using segsum to avoid catastrophic cancellation
    L = segsum(A)  # L[i,j] = sum(A[j:i+1]) = log(prod(exp(A[j:i+1])))

    # Compute state evolution with log-space products
    # ... (rest of SSD algorithm)
```

---

## Part 3: Applying to Elman/Triple R

### 3.1 Current Triple R Dynamics

```python
# Current implementation (mingru/haste_elman_triple_r.py)
def forward(x, h_prev, R_h, R_x, R_delta, W_delta, b, b_delta):
    # Input-dependent delta (gate)
    delta_raw = h_prev @ R_delta + x @ W_delta + b_delta
    delta = torch.sigmoid(delta_raw)  # [0, 1]

    # Candidate computation
    candidate_raw = h_prev @ R_h + x @ R_x + b
    candidate = torch.tanh(candidate_raw)

    # State update (linear interpolation)
    h_new = (1 - delta) * h_prev + delta * candidate

    return h_new
```

### 3.2 Why Full R Blocks Log-Space

The term `h_prev @ R_h` is a matrix-vector product. You cannot take the log of this because:

1. R_h has positive and negative entries
2. The result can be negative
3. log(negative) is undefined

Log-space only works for **positive multiplicative factors**.

### 3.3 Option A: Diagonal Elman for Log-Space

**This is the simpler approach** - convert to diagonal transitions like Mamba2. See Part 8.6 for the more complex Option B that preserves full R matrices.

```python
# Diagonal version
def forward_diagonal(x, h_prev, r_h, r_x, r_delta, W_delta, b, b_delta):
    """
    r_h: (d_model,) - diagonal decay per dimension
    r_x: (d_model,) - diagonal input mixing
    r_delta: (d_model,) - diagonal delta modulation
    """
    # Delta gate (keep full W_delta for input path - not recurrent)
    delta_raw = h_prev * r_delta + x @ W_delta + b_delta
    delta = torch.sigmoid(delta_raw)

    # Candidate (diagonal transitions)
    candidate_raw = h_prev * r_h + x * r_x + b
    candidate = torch.tanh(candidate_raw)

    # State update
    h_new = (1 - delta) * h_prev + delta * candidate

    return h_new
```

### 3.4 Log-Space Diagonal Elman

Now we can implement log-space computation:

```python
def forward_logspace(x, log_h_prev, r_h, r_x, r_delta, W_delta, b, b_delta):
    """
    log_h_prev: (batch, d_model) - hidden state in LOG SPACE
    r_h: (d_model,) - decay factors, must be POSITIVE (we'll use softplus)

    Key insight: Represent h in log-space to prevent underflow.
    h = exp(log_h), so log(h_new) = log(decay * h + input)
    """
    # Convert from log-space for operations that need linear values
    h_prev = torch.exp(log_h_prev)  # Back to linear for tanh etc.

    # Delta gate (linear space - sigmoid naturally bounded)
    delta_raw = h_prev * r_delta + x @ W_delta + b_delta
    delta = torch.sigmoid(delta_raw)

    # For the recurrent part, work in log-space
    # h_new = (1 - delta) * h_prev + delta * tanh(...)

    # Decay term: (1 - delta) * h_prev
    # log((1-delta) * h) = log(1-delta) + log(h)
    log_decay_term = log_one_minus_sigmoid(delta_raw) + log_h_prev

    # Candidate term: delta * tanh(h_prev * r_h + x * r_x + b)
    candidate_raw = h_prev * r_h + x * r_x + b
    candidate = torch.tanh(candidate_raw)
    # For tanh output in [-1, 1], need to handle sign
    # Split into positive and negative parts
    log_candidate = log_safe_abs(candidate)
    sign_candidate = torch.sign(candidate)
    log_input_term = log_sigmoid(delta_raw) + log_candidate

    # Combine: log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
    # This is logaddexp, but we need to handle signs
    log_h_new = logaddexp_signed(log_decay_term, log_input_term, sign_candidate)

    return log_h_new
```

### 3.5 The Sign Problem

tanh outputs values in [-1, 1], which breaks pure log-space. Solutions:

**Option A: Split representation**
```python
# Store (log_abs_h, sign_h) separately
log_abs_h = torch.log(torch.abs(h) + eps)
sign_h = torch.sign(h)
```

**Option B: Use softplus activation instead of tanh**
```python
# softplus(x) > 0 always, can use log-space directly
candidate = F.softplus(candidate_raw)
log_candidate = torch.log(candidate)
```

**Option C: Mamba2's approach - use exp for A, keep B/C linear**
```python
# A (decay) is log-space: A = -softplus(A_raw), exp(A) in (0, 1)
# B, C are linear - only the recurrence is log-space
```

Mamba2 uses Option C: The state decay is log-space, but input/output projections are linear.

---

## Part 4: Implementation Plan

### 4.1 File Structure

```
mingru/
├── logspace_elman.py          # New: Log-space Elman with diagonal transitions
├── logspace_elman_triton.py   # New: Triton kernel for fused log-space ops
├── segsum.py                  # New: Segment-sum implementation
└── diagonal_mhtr.py           # Existing: Can be extended with log-space
```

### 4.2 Core Functions to Implement

#### 4.2.1 Numerically Stable Log Helpers

```python
def log_sigmoid(x):
    """log(sigmoid(x)) = -softplus(-x)"""
    return -F.softplus(-x)

def log_one_minus_sigmoid(x):
    """log(1 - sigmoid(x)) = -softplus(x)"""
    return -F.softplus(x)

def logaddexp_signed(log_a, log_b, sign_b):
    """
    Compute log(|a + sign_b * b|) where a > 0.

    For sign_b = +1: log(a + b) = log(a) + log(1 + exp(log_b - log_a))
    For sign_b = -1: log(a - b) = log(a) + log(1 - exp(log_b - log_a))
                                = log(a) + log(-expm1(log_b - log_a)) if b < a

    Returns: (log_abs_result, sign_result)
    """
    # Implementation needed - handle all sign cases
    pass
```

#### 4.2.2 Segment Sum

```python
def segsum(x, dim=-1):
    """
    Compute segment cumulative sums for log-space products.

    Input: x of shape (..., T)
    Output: matrix of shape (..., T, T) where result[..., i, j] = sum(x[..., j:i+1])

    For numerical stability, this replaces the naive:
        cumsum[i] - cumsum[j-1]
    which causes catastrophic cancellation.
    """
    T = x.shape[dim]
    # Create indices for segment computation
    # ... implementation
    pass

def segsum_chunked(x, chunk_size=64):
    """
    Chunked version for memory efficiency.
    Computes full segsum within chunks, then propagates across chunks.
    """
    pass
```

#### 4.2.3 Log-Space Elman Cell

```python
class LogSpaceElmanCell(nn.Module):
    """
    Single Elman cell with log-space hidden state computation.

    Key differences from standard Elman:
    1. Diagonal transitions (r_h is vector, not matrix R_h)
    2. Hidden state stored in log-space
    3. Decay computed via log cumsum, not direct multiplication
    """

    def __init__(self, dim, delta_init=-2.0):
        super().__init__()
        # Diagonal decay - initialized near 1 (log near 0)
        self.r_h = nn.Parameter(torch.zeros(dim))  # log-space decay

        # Input mixing - can be full matrix since not recurrent
        self.W_x = nn.Linear(dim, dim, bias=False)

        # Delta (gate) computation
        self.r_delta = nn.Parameter(torch.zeros(dim))
        self.W_delta = nn.Linear(dim, dim, bias=False)
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Bias
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x, log_h_prev):
        """
        x: (batch, dim) - input in linear space
        log_h_prev: (batch, dim) - previous hidden in LOG space

        Returns: log_h_new in LOG space
        """
        # Get linear h for gate computation
        h_prev = torch.exp(log_h_prev.clamp(max=20))  # Prevent overflow

        # Delta gate
        delta_raw = h_prev * self.r_delta + self.W_delta(x) + self.b_delta
        delta = torch.sigmoid(delta_raw)
        log_delta = log_sigmoid(delta_raw)
        log_one_minus_delta = log_one_minus_sigmoid(delta_raw)

        # Decay contribution: (1 - delta) * h_prev
        # In log-space: log(1-delta) + log(h_prev)
        log_decay = log_one_minus_delta + log_h_prev

        # Input contribution: delta * activation(...)
        candidate_raw = h_prev * torch.exp(self.r_h) + self.W_x(x) + self.b
        # Use softplus to keep positive (enables log-space)
        candidate = F.softplus(candidate_raw)
        log_candidate = torch.log(candidate + 1e-8)
        log_input = log_delta + log_candidate

        # Combine in log-space: log(exp(log_decay) + exp(log_input))
        log_h_new = torch.logaddexp(log_decay, log_input)

        return log_h_new
```

### 4.3 Triton Kernel for Fused Operations

For efficiency, fuse the log-space operations:

```python
@triton.jit
def logspace_elman_fwd_kernel(
    # Input pointers
    x_ptr, log_h_ptr, r_h_ptr, r_delta_ptr, w_delta_ptr, w_x_ptr, b_ptr, b_delta_ptr,
    # Output pointer
    log_h_new_ptr,
    # Dimensions
    batch, dim,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    """Fused log-space Elman forward pass."""
    pid = tl.program_id(0)

    # Load inputs
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    x = tl.load(x_ptr + offs, mask=mask)
    log_h = tl.load(log_h_ptr + offs, mask=mask)
    r_h = tl.load(r_h_ptr + offs, mask=mask)
    r_delta = tl.load(r_delta_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    b_delta = tl.load(b_delta_ptr + offs, mask=mask)

    # Compute h_prev from log
    h_prev = tl.exp(tl.minimum(log_h, 20.0))

    # Delta computation (need W_delta @ x - handle separately)
    # ... simplified for diagonal case
    delta_raw = h_prev * r_delta + b_delta

    # Log-space sigmoid
    log_delta = -tl.log(1.0 + tl.exp(-delta_raw))
    log_one_minus_delta = -tl.log(1.0 + tl.exp(delta_raw))

    # Decay term
    log_decay = log_one_minus_delta + log_h

    # Input term (softplus activation)
    candidate_raw = h_prev * tl.exp(r_h) + x + b
    candidate = tl.log(1.0 + tl.exp(candidate_raw))  # softplus
    log_candidate = tl.log(candidate + 1e-8)
    log_input = log_delta + log_candidate

    # Logaddexp
    max_val = tl.maximum(log_decay, log_input)
    log_h_new = max_val + tl.log(tl.exp(log_decay - max_val) + tl.exp(log_input - max_val))

    # Store
    tl.store(log_h_new_ptr + offs, log_h_new, mask=mask)
```

### 4.4 Integration with Existing Architecture

```python
class LogSpaceElmanLM(nn.Module):
    """
    Language model using log-space Elman layers.
    Drop-in replacement for existing Elman variants.
    """

    def __init__(self, num_tokens, dim, depth, ...):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([
            LogSpaceElmanLayer(dim, ...)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight  # Tie weights

    def forward(self, x, prev_hiddens=None, ...):
        B, T = x.shape

        # Embed (linear space)
        h = self.token_emb(x)

        # Initialize log-space hidden states
        if prev_hiddens is None:
            log_hiddens = [torch.zeros(B, self.dim, device=x.device) for _ in range(self.depth)]
        else:
            log_hiddens = prev_hiddens

        new_log_hiddens = []

        for layer_idx, layer in enumerate(self.layers):
            # Layer processes sequence, returns updated log-hidden
            h, log_h_new = layer(h, log_hiddens[layer_idx])
            new_log_hiddens.append(log_h_new)

        # Output (linear space)
        h = self.norm(h)
        logits = self.to_logits(h)

        return logits, new_log_hiddens
```

---

## Part 5: Testing and Validation

### 5.1 Numerical Stability Tests

```python
def test_no_nan_deep_forward():
    """Verify no NaN through 32 layers, 1024 tokens."""
    model = LogSpaceElmanLM(num_tokens=50281, dim=2048, depth=32)
    x = torch.randint(0, 50281, (4, 1024))

    with torch.no_grad():
        logits, _ = model(x)

    assert not torch.isnan(logits).any(), "NaN in forward pass"
    assert not torch.isinf(logits).any(), "Inf in forward pass"

def test_gradient_flow():
    """Verify gradients don't vanish through deep network."""
    model = LogSpaceElmanLM(num_tokens=50281, dim=2048, depth=32)
    x = torch.randint(0, 50281, (4, 512))

    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, 50281), x.view(-1))
    loss.backward()

    # Check gradient norms per layer
    for i, layer in enumerate(model.layers):
        grad_norm = layer.cell.r_h.grad.norm().item()
        print(f"Layer {i}: grad_norm = {grad_norm:.6f}")
        assert grad_norm > 1e-8, f"Vanishing gradient at layer {i}"

def test_matches_linear_space():
    """Verify log-space gives same results as linear-space for short sequences."""
    # Compare LogSpaceElman vs standard Elman for 10 tokens
    # Should match to numerical precision
    pass
```

### 5.2 Training Comparison

```bash
# Run log-space Elman at 1.3B, 1000 steps
# Compare to:
# - Mamba2: 3.924 avg50 (target)
# - Triple R: 4.216 avg50 (current best)

./train.logspace_elman_1b.sh
```

### 5.3 Metrics to Track

1. **Loss curve**: Should match or beat Triple R
2. **Hidden state norms**: Track per-layer to verify no explosion/collapse
3. **Gradient norms**: Should be stable across layers (unlike current gradient death)
4. **Throughput**: Log-space ops may be slower - measure tok/s

---

## Part 6: Expected Outcomes

### 6.1 If Log-Space Works

- Loss should match or approach Mamba2 (3.9 vs 4.2)
- Gradient death should be eliminated
- Can train for 10k+ steps without collapse
- Diagonal constraint is acceptable trade-off

### 6.2 If Log-Space Doesn't Help

- Mamba2's advantage is in SSD parallelization, not numerics
- Need to investigate other factors:
  - Multi-head structure
  - Specific initialization
  - The C (output) projection

### 6.3 Hybrid Approaches to Try

1. **Log-space for deep layers only**: Use standard Elman for layers 1-10, log-space for 11-32
2. **Periodic normalization**: Reset log-space every N layers
3. **Mixed precision**: Log-space in fp32, linear computations in bf16

---

## Part 7: References

### Papers
- Mamba-2 (SSD): Dao & Gu, 2024, arXiv:2405.21060
- Mamba: Gu & Dao, 2023, arXiv:2312.00752
- Log-space RNNs: Various numerical stability literature

### Code References
- Mamba2 minimal SSD: `mamba-ssm/mamba_ssm/modules/ssd_minimal.py`
- Our log-space GRU: `mingru/logspace_hybrid_gru.py`
- Diagonal MHTR: `mingru/diagonal_mhtr.py`

### Blog Posts
- [SSD Algorithm (Tri Dao)](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
- [SSD Theory (Goomba Lab)](https://goombalab.github.io/blog/2024/mamba2-part2-theory/)

---

## Part 8: Preserving What Works

### 8.1 Output Selectivity Still Matters

**Important distinction**: Our experiments showed that **input selectivity** (Mamba2's B gate) has no effect on Triple R. But **output selectivity** (compete×silu) is load-bearing.

```python
# Input selectivity (NO effect when added to Triple R):
B_gate = sigmoid(W_B @ x)  # Mamba2's B gate
state_input = B_gate * x   # Modulates what enters state

# Output selectivity (DOES help, keep this):
compete = softmax(h, groups=32)  # Group competition
output = compete * silu(W_out @ h)  # Selective output gating
```

**For log-space implementation**: Output selectivity happens AFTER the recurrent computation, so it's unaffected by log-space dynamics. Keep the compete×silu output gate exactly as-is.

```python
class LogSpaceElmanLayer(nn.Module):
    def forward(self, x, log_h_prev):
        # Log-space recurrence (new)
        log_h = self.log_space_cell(x, log_h_prev)

        # Convert to linear for output
        h = torch.exp(log_h.clamp(max=20))

        # Output selectivity (unchanged from current implementation)
        compete = F.softmax(h.view(B, T, self.n_groups, -1), dim=-1)
        compete = compete.view(B, T, -1)
        output = compete * F.silu(self.out_proj(h))

        return output, log_h
```

### 8.2 Full Matrices Where They Help

The diagonal constraint only applies to the **recurrent path** (h_{t-1} → h_t). All non-recurrent projections can remain full matrices:

```python
# MUST be diagonal (recurrent, log-space critical):
decay_term = log_h_prev + log_r_h  # Element-wise: r_h is vector

# CAN be full matrix (non-recurrent):
input_proj = x @ W_x           # Input to hidden
delta_input = x @ W_delta      # Input to gate
output_proj = h @ W_out        # Hidden to output
```

**Architecture preserving full-matrix power**:

```
Input x ──────────────────────────────────────────> [Full W_x] ──────┐
                                                                      │
                                                                      v
Log-Hidden ──> [Diag r_h decay] ──> logaddexp ──> New Log-Hidden ────┘
                                         ^                │
                                         │                v
Gate: [Full W_delta] @ x + r_delta * h ──┘         [Full W_out] @ exp(h)
                                                          │
                                                          v
                                                   Output selectivity
                                                   (compete × silu)
```

This keeps **3 full matrices** (W_x, W_delta, W_out) while only the recurrent decay is diagonal.

### 8.3 Preserving tanh with Signed Log-Space

tanh is important for expressivity - it provides bounded nonlinearity that prevents explosion. We CAN preserve it using **signed log representation**:

```python
def signed_log(x):
    """Convert to (log_abs, sign) representation."""
    sign = torch.sign(x)
    log_abs = torch.log(torch.abs(x) + 1e-8)
    return log_abs, sign

def signed_exp(log_abs, sign):
    """Convert back from (log_abs, sign) to linear."""
    return sign * torch.exp(log_abs)
```

**Full log-space cell with tanh**:

```python
class LogSpaceElmanCellWithTanh(nn.Module):
    """
    Log-space Elman with tanh activation.

    State representation: (log_abs_h, sign_h)
    - log_abs_h: log(|h|), always valid
    - sign_h: {-1, +1}, tracks sign of h
    """

    def forward(self, x, log_abs_h_prev, sign_h_prev):
        # Get linear h for computations
        h_prev = sign_h_prev * torch.exp(log_abs_h_prev.clamp(max=20))

        # Delta gate (linear space, bounded by sigmoid)
        delta_raw = h_prev * self.r_delta + self.W_delta(x) + self.b_delta
        delta = torch.sigmoid(delta_raw)

        # Candidate with TANH (linear space)
        candidate_raw = h_prev * torch.exp(self.r_h) + self.W_x(x) + self.b
        candidate = torch.tanh(candidate_raw)  # [-1, 1]

        # Decay term: (1 - delta) * h_prev
        decay = (1 - delta) * h_prev  # Can be negative

        # Input term: delta * candidate
        input_term = delta * candidate  # Can be negative

        # New state: h_new = decay + input_term
        h_new = decay + input_term

        # Convert to signed log representation
        sign_h_new = torch.sign(h_new)
        log_abs_h_new = torch.log(torch.abs(h_new) + 1e-8)

        return log_abs_h_new, sign_h_new
```

**Why this works**:
1. All arithmetic happens in linear space (tanh, addition preserved)
2. Result converted to log for storage (prevents underflow accumulation)
3. Sign tracked separately (handles negative values)
4. logaddexp not needed - we just do linear arithmetic then convert

**The key insight**: We don't need log-space for the arithmetic, just for **storing** the hidden state between steps. This prevents the multiplicative underflow across thousands of steps.

**⚠️ Limitation**: This is "log-space storage", not true "log-space computation". The arithmetic still happens in linear space:

```python
h_prev = sign * exp(log_h_prev)  # Convert to linear (can underflow!)
# ... linear arithmetic ...
log_h_new = log(|h_new|)         # Convert back to log
```

If `log_h_prev` is very negative (e.g., -100), then `exp(-100) ≈ 0` in float32 anyway. The log-space storage only helps if we **never** convert back to linear.

For true stability, the logsumexp approach in Part 8.6 is needed - it keeps everything in log-space until the final output.

### 8.4 Numerical Stability of Signed Log Approach

The signed log approach has one vulnerability: when h ≈ 0, log(|h|) → -∞.

**Solution: Clamped log representation**

```python
def safe_signed_log(x, min_abs=1e-8, max_log=40):
    """
    Numerically stable signed log.

    Clamps |x| to [min_abs, exp(max_log)] to prevent:
    - log(0) = -inf
    - exp(large) = inf
    """
    sign = torch.sign(x)
    # For zero, sign is 0, treat as positive epsilon
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    abs_x = torch.abs(x).clamp(min=min_abs)
    log_abs = torch.log(abs_x).clamp(max=max_log)

    return log_abs, sign
```

### 8.5 Architecture Summary: Best of Both Worlds

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOG-SPACE ELMAN WITH TANH                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stored State: (log_abs_h, sign_h) - prevents underflow            │
│                                                                     │
│  Recurrence: DIAGONAL r_h (enables log storage, stable decay)      │
│                                                                     │
│  Activation: TANH (preserved! computed in linear space)            │
│                                                                     │
│  Input Path: FULL MATRIX W_x (expressivity preserved)              │
│                                                                     │
│  Gate Path: FULL MATRIX W_delta (expressivity preserved)           │
│                                                                     │
│  Output: FULL MATRIX W_out + compete×silu (selectivity preserved)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

What changes: Only recurrent h→h transition becomes diagonal
What stays: tanh, full matrices, output selectivity, compete×silu
Benefit: Numerical stability at depth without sacrificing expressivity
```

### 8.6 Log-Space Matrix Multiplication (Full R Preserved!)

**Actually, we CAN do log-space matrix-vector products.** The key is decomposing R into positive and negative parts and using logsumexp.

#### The Math

For R @ h where h is in log-space (log_h = log(|h|), sign_h = sign(h)):

```
(R @ h)_j = Σ_i R_{ji} * h_i
          = Σ_i R_{ji} * sign_h_i * exp(log_h_i)
```

Decompose R into positive/negative parts:
```python
R_pos = R.clamp(min=0)      # R+ = max(R, 0)
R_neg = (-R).clamp(min=0)   # R- = max(-R, 0), so R = R+ - R-
```

Then:
```
(R @ h)_j = Σ_i R+_{ji} * sign_h_i * exp(log_h_i)
          - Σ_i R-_{ji} * sign_h_i * exp(log_h_i)
```

Each sum splits into positive-signed and negative-signed terms:
```
positive_contribution = Σ_{i: sign_h_i > 0} R+_{ji} * exp(log_h_i)
                      + Σ_{i: sign_h_i < 0} R-_{ji} * exp(log_h_i)

negative_contribution = Σ_{i: sign_h_i > 0} R-_{ji} * exp(log_h_i)
                      + Σ_{i: sign_h_i < 0} R+_{ji} * exp(log_h_i)
```

In log-space, each sum is a **logsumexp**:
```python
log_pos = logsumexp(log_R + log_h, dim=-1)  # Over input dimension
```

#### Implementation

```python
class LogSpaceMatmul(nn.Module):
    """
    Log-space matrix-vector multiplication.
    Preserves full R matrix while working in log-space.
    """

    def __init__(self, dim):
        super().__init__()
        self.R = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, log_h, sign_h):
        """
        log_h: [B, D] - log(|h|)
        sign_h: [B, D] - sign(h) in {-1, +1}

        Returns: log_result, sign_result for R @ h
        """
        B, D = log_h.shape

        # Decompose R into positive and negative parts (can cache these)
        R_pos = self.R.clamp(min=0)
        R_neg = (-self.R).clamp(min=0)

        log_R_pos = torch.log(R_pos + 1e-10)  # [D, D]
        log_R_neg = torch.log(R_neg + 1e-10)

        # Mask for positive and negative h values
        pos_mask = (sign_h > 0).float()  # [B, D]
        neg_mask = (sign_h < 0).float()

        # For positive contribution to result:
        # = R+ @ (pos_h) + R- @ (neg_h)
        # log version: logsumexp over input dim

        # [B, D_out, D_in] = [1, D, D] + [B, 1, D]
        log_terms_pp = log_R_pos.unsqueeze(0) + log_h.unsqueeze(1)  # R+ * h where h>0
        log_terms_pn = log_R_neg.unsqueeze(0) + log_h.unsqueeze(1)  # R- * h where h<0

        # Mask to only include positive-h terms for pp, negative-h for pn
        log_terms_pp = log_terms_pp + torch.log(pos_mask + 1e-10).unsqueeze(1)
        log_terms_pn = log_terms_pn + torch.log(neg_mask + 1e-10).unsqueeze(1)

        # Positive contribution = logsumexp of valid terms
        log_pos_contrib = torch.logsumexp(
            torch.stack([log_terms_pp, log_terms_pn], dim=0),
            dim=(0, -1)
        )  # [B, D_out]

        # Similarly for negative contribution
        log_terms_np = log_R_neg.unsqueeze(0) + log_h.unsqueeze(1)  # R- * h where h>0
        log_terms_nn = log_R_pos.unsqueeze(0) + log_h.unsqueeze(1)  # R+ * h where h<0

        log_terms_np = log_terms_np + torch.log(pos_mask + 1e-10).unsqueeze(1)
        log_terms_nn = log_terms_nn + torch.log(neg_mask + 1e-10).unsqueeze(1)

        log_neg_contrib = torch.logsumexp(
            torch.stack([log_terms_np, log_terms_nn], dim=0),
            dim=(0, -1)
        )

        # Result = pos_contrib - neg_contrib
        # Convert to signed log representation
        pos_val = torch.exp(log_pos_contrib.clamp(max=40))
        neg_val = torch.exp(log_neg_contrib.clamp(max=40))
        result = pos_val - neg_val

        sign_result = torch.sign(result)
        log_result = torch.log(torch.abs(result) + 1e-10)

        return log_result, sign_result
```

#### Triton Kernel (Efficient Implementation)

```python
@triton.jit
def log_matmul_kernel(
    # Inputs
    log_h_ptr,      # [B, D_in] log(|h|)
    sign_h_ptr,     # [B, D_in] sign(h)
    log_R_pos_ptr,  # [D_out, D_in] log(max(R, 0))
    log_R_neg_ptr,  # [D_out, D_in] log(max(-R, 0))
    # Outputs
    log_out_ptr,    # [B, D_out]
    sign_out_ptr,   # [B, D_out]
    # Dims
    B, D_out, D_in,
    # Block sizes
    BLOCK_B: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """Fused log-space matrix-vector product."""
    pid_b = tl.program_id(0)
    pid_out = tl.program_id(1)

    # Accumulate logsumexp for positive and negative contributions
    # Use online logsumexp algorithm for numerical stability

    max_pos = float('-inf')
    sum_pos = 0.0
    max_neg = float('-inf')
    sum_neg = 0.0

    for i_start in range(0, D_in, BLOCK_IN):
        i_offs = i_start + tl.arange(0, BLOCK_IN)
        mask = i_offs < D_in

        # Load inputs
        log_h = tl.load(log_h_ptr + pid_b * D_in + i_offs, mask=mask, other=float('-inf'))
        sign_h = tl.load(sign_h_ptr + pid_b * D_in + i_offs, mask=mask, other=0.0)

        # Load R values for this output dimension
        log_R_p = tl.load(log_R_pos_ptr + pid_out * D_in + i_offs, mask=mask, other=float('-inf'))
        log_R_n = tl.load(log_R_neg_ptr + pid_out * D_in + i_offs, mask=mask, other=float('-inf'))

        # Compute log terms
        # Positive contrib: R+ where h>0, R- where h<0
        log_term_pos = tl.where(sign_h > 0, log_R_p + log_h, log_R_n + log_h)
        # Negative contrib: R- where h>0, R+ where h<0
        log_term_neg = tl.where(sign_h > 0, log_R_n + log_h, log_R_p + log_h)

        # Online logsumexp update
        new_max_pos = tl.maximum(max_pos, tl.max(log_term_pos, axis=0))
        sum_pos = sum_pos * tl.exp(max_pos - new_max_pos) + tl.sum(tl.exp(log_term_pos - new_max_pos), axis=0)
        max_pos = new_max_pos

        new_max_neg = tl.maximum(max_neg, tl.max(log_term_neg, axis=0))
        sum_neg = sum_neg * tl.exp(max_neg - new_max_neg) + tl.sum(tl.exp(log_term_neg - new_max_neg), axis=0)
        max_neg = new_max_neg

    # Final logsumexp
    log_pos_contrib = max_pos + tl.log(sum_pos)
    log_neg_contrib = max_neg + tl.log(sum_neg)

    # Compute result = exp(log_pos) - exp(log_neg)
    pos_val = tl.exp(tl.minimum(log_pos_contrib, 40.0))
    neg_val = tl.exp(tl.minimum(log_neg_contrib, 40.0))
    result = pos_val - neg_val

    sign_out = tl.where(result >= 0, 1.0, -1.0)
    log_out = tl.log(tl.abs(result) + 1e-10)

    # Store
    tl.store(log_out_ptr + pid_b * D_out + pid_out, log_out)
    tl.store(sign_out_ptr + pid_b * D_out + pid_out, sign_out)
```

#### Complexity Analysis

| Operation | Standard | Log-Space |
|-----------|----------|-----------|
| R @ h | O(D²) matmul | O(D²) logsumexp |
| Memory | O(D²) for R | O(2D²) for log_R_pos, log_R_neg |
| Numerical stability | Underflow at depth | Stable via logsumexp |

The overhead is ~2x memory for storing decomposed R, and some extra ops for logsumexp. But we get **full R matrices with log-space stability**.

#### Integration with Triple R

```python
class LogSpaceTripleR(nn.Module):
    """
    Full Triple R architecture with log-space numerics.

    Preserves:
    - Full R_h, R_x, R_delta matrices (not diagonal!)
    - tanh activation
    - compete×silu output gate

    Gains:
    - Log-space storage prevents underflow
    - Stable training at 32+ layers
    """

    def __init__(self, dim, d_state_r=16, delta_init=-2.0):
        super().__init__()
        # Full R matrices (can use low-rank if desired)
        self.log_R_h = LogSpaceMatmul(dim)
        self.log_R_x = LogSpaceMatmul(dim)
        self.log_R_delta = LogSpaceMatmul(dim)

        # Gate input projection
        self.W_delta = nn.Linear(dim, dim, bias=False)
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))
        self.b = nn.Parameter(torch.zeros(dim))

        # Output
        self.out_proj = nn.Linear(dim, dim)
        self.n_groups = 32

    def forward(self, x, log_h_prev, sign_h_prev):
        B, D = x.shape

        # Convert h from log-space for gate computation
        h_prev = sign_h_prev * torch.exp(log_h_prev.clamp(max=20))

        # Delta gate
        delta_raw = h_prev @ self.log_R_delta.R + self.W_delta(x) + self.b_delta
        delta = torch.sigmoid(delta_raw)

        # Candidate in log-space: R_h @ h + R_x @ x + b
        # For tanh, compute in linear space then convert back

        log_Rh_h, sign_Rh_h = self.log_R_h(log_h_prev, sign_h_prev)
        Rh_h = sign_Rh_h * torch.exp(log_Rh_h.clamp(max=20))

        Rx_x = x @ self.log_R_x.R  # Input path doesn't need log-space (x is fresh)

        candidate_raw = Rh_h + Rx_x + self.b
        candidate = torch.tanh(candidate_raw)

        # State update
        h_new = (1 - delta) * h_prev + delta * candidate

        # Convert to signed log for storage
        sign_h_new = torch.sign(h_new)
        sign_h_new = torch.where(sign_h_new == 0, torch.ones_like(sign_h_new), sign_h_new)
        log_h_new = torch.log(torch.abs(h_new) + 1e-10).clamp(max=40)

        # Output with selectivity
        compete = F.softmax(h_new.view(B, self.n_groups, -1), dim=-1).view(B, D)
        output = compete * F.silu(self.out_proj(h_new))

        return output, log_h_new, sign_h_new
```

### 8.7 Summary: We Can Have It All

With log-space matrix multiplication:

| Feature | Status |
|---------|--------|
| Full R matrices | ✅ Preserved via logsumexp decomposition |
| tanh activation | ✅ Preserved with signed log |
| Output selectivity | ✅ Preserved (compete×silu) |
| Numerical stability | ✅ Log-space storage + logsumexp |
| Custom Haste kernel | ✅ Triton implementation ready |

**The only trade-off**: ~2x overhead for decomposed R storage and logsumexp ops. Worth it for deep training stability.

### 8.8 Comparison to Mamba2

| Component | Mamba2 | Log-Space Triple R |
|-----------|--------|-----------------|
| Recurrent transition | Diagonal A | **Full R matrix** (log-space matmul) |
| Input projection | B matrix (selective) | Full R_x matrix |
| Output projection | C matrix | Full W_out + **compete×silu** |
| Activation | Linear in state | **tanh** (nonlinear) |
| State storage | Log-space | Signed log-space |
| Numerical stability | segsum | logsumexp decomposition |

**We get MORE expressivity than Mamba2** while matching its numerical stability.

---

## Part 9: Principled Ablation Ladder

Build up systematically from stock Elman, measuring the impact of each modification.

### The Ladder: Stock Elman → Triple R

```
Level 0: Stock Elman
    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)

    ↓ Add: Input-dependent gate (GRU-style delta)

Level 1: Gated Elman
    delta = sigmoid(W_delta @ x_t + b_delta)
    h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + W_h @ h_{t-1} + b)

    ↓ Add: Output selectivity (compete×silu)

Level 2: Selective Elman
    [same recurrence as Level 1]
    output = compete(h_t) * silu(W_out @ h_t)

    ↓ Change: Diagonal W_h → vector r_h (Mamba2-style)

Level 3: Diagonal Selective Elman
    h_t = (1 - delta) * h_{t-1} + delta * tanh(W_x @ x_t + r_h * h_{t-1} + b)

    ↓ Add: Log-space storage (signed log representation)

Level 4: Log-Storage Diagonal Elman
    Store (log|h|, sign(h)) instead of h
    Still compute in linear space, convert at boundaries

    ↓ Change: Full W_h with logsumexp (restore expressivity)

Level 5: Log-Compute Full Elman
    Use logsumexp decomposition for W_h @ h
    Full matrix expressivity + numerical stability

    ↓ Add: Per-dimension delta modulation (R_delta)

Level 6: Triple R
    delta_mod = h_{t-1} @ R_delta
    h_t = (1 - delta) * h_{t-1} + delta * tanh(x @ R_x + h_{t-1} @ R_h + delta_mod + b)
```

### Experiment Matrix

| Level | Model | Key Addition | Expected Impact | Params (1.3B) |
|-------|-------|--------------|-----------------|---------------|
| 0 | Stock Elman | — | Baseline | ~1.0B |
| 1 | Gated Elman | Input-dependent delta | +Better gradient flow | ~1.1B |
| 2 | Selective Elman | compete×silu output | +Implicit sparsity | ~1.2B |
| 3 | Diagonal Selective | Diagonal W_h | +Stability? -Expressivity? | ~1.1B |
| 4 | Log-Storage Diagonal | Log storage | +Numerical stability | ~1.1B |
| 5 | Log-Compute Full | Logsumexp matmul | +Expressivity back | ~1.2B |
| 6 | Triple R | R_delta modulation | +Richer gating | ~1.3B |

### What Each Level Tests

**Level 0 → 1**: Does input-dependent gating help?
- Hypothesis: Yes, allows model to control memory/update tradeoff per position
- If NO improvement: Gating not essential, focus elsewhere

**Level 1 → 2**: Does output selectivity help?
- Hypothesis: Yes, compete×silu provides implicit sparsity + smooth gradients
- If NO improvement: Can simplify output path

**Level 2 → 3**: Does diagonal hurt expressivity?
- Hypothesis: Slight degradation but more stable
- If SIGNIFICANT degradation: Full matrices matter, need logsumexp path
- If NO degradation: Diagonal is fine, simpler is better

**Level 3 → 4**: Does log storage help stability?
- Hypothesis: Helps prevent underflow in very deep/long training
- Diagnostic: Track hidden state magnitudes before/after
- If NO improvement: Problem isn't numerical underflow

**Level 4 → 5**: Can we recover full-matrix expressivity?
- Hypothesis: Logsumexp gives best of both worlds
- Trade-off: ~2x compute overhead
- If overhead too high: Stick with diagonal

**Level 5 → 6**: Does R_delta add value?
- Hypothesis: Per-dimension gate modulation helps
- If NO improvement: Simpler is better, stop at Level 5

### Comparison Points

At each level, compare to:
- **Mamba2** (3.924 avg50 @ 1000 steps) - Target to match
- **Previous level** - Isolate effect of each modification
- **Diagonal MHTR** (running) - Alternative approach

### Implementation Order

```python
# Run these in sequence, 1000 steps each, 1.3B params

experiments = [
    ("stock_elman",           {"diagonal": False, "log_space": False, "gate": False, "output_select": False}),
    ("gated_elman",           {"diagonal": False, "log_space": False, "gate": True,  "output_select": False}),
    ("selective_elman",       {"diagonal": False, "log_space": False, "gate": True,  "output_select": True}),
    ("diagonal_selective",    {"diagonal": True,  "log_space": False, "gate": True,  "output_select": True}),
    ("log_storage_diagonal",  {"diagonal": True,  "log_space": "storage", "gate": True,  "output_select": True}),
    ("log_compute_full",      {"diagonal": False, "log_space": "compute", "gate": True,  "output_select": True}),
    ("triple_r",              {"diagonal": False, "log_space": "compute", "gate": True,  "output_select": True, "r_delta": True}),
]
```

### Expected Outcomes

**Best case**: One of the middle levels matches Mamba2, revealing the key ingredient
**Worst case**: Even Level 6 (full Triple R with log-space) doesn't match → Problem is elsewhere

### Key Metrics Per Level

1. **avg50 loss** - Primary comparison
2. **tok/s** - Throughput cost of each addition
3. **hidden state stats** - Mean, max, min of |h| per layer
4. **gradient norms** - Per layer, detect vanishing

---

## Part 10: Detailed Implementation Plan

Given the uncertainties, implement incrementally:

### Phase 1: Validate the Hypothesis (Before Writing Custom Kernels)

1. **Wait for Diagonal MHTR results** - If diagonal structure alone matches Mamba2, log-space may not be needed
2. **Profile hidden state magnitudes** - Track `|h|` statistics through training to see if underflow is actually occurring
3. **Test simple fixes first**:
   - LayerNorm on hidden state each step
   - Gradient clipping per-layer
   - Hidden state clipping to [-10, 10]

```python
# Quick diagnostic: add to training loop
if step % 100 == 0:
    for i, layer in enumerate(model.layers):
        h = layer.last_hidden  # Need to save this
        print(f"Layer {i}: h_mean={h.abs().mean():.6f}, h_max={h.abs().max():.6f}")
```

### Phase 2: Simple Log-Space Storage (If Needed)

If Phase 1 shows hidden states collapsing:

1. Implement signed log storage (Section 8.3) - minimal code change
2. Test at same scale (1.3B, 1000 steps)
3. Compare to baseline

### Phase 3: Full Log-Space Matmul (If Storage Isn't Enough)

If Phase 2 doesn't help:

1. Implement logsumexp matmul (Section 8.6)
2. First in pure PyTorch (slow but correct)
3. Validate numerics match standard matmul on small examples
4. Then write Triton kernel

### Phase 4: Triton Optimization

Only after PyTorch version works:

1. Port to Triton with fused ops
2. Benchmark throughput vs standard matmul
3. If >3x slower, consider diagonal fallback

### Decision Tree

```
Diagonal MHTR matches Mamba2?
├─ YES → Diagonal structure is key, not log-space
│        → Use DiagonalMHTR, skip log-space work
│
└─ NO → Check hidden state magnitudes
        ├─ States collapsing → Try log-space storage
        │   ├─ Storage helps → Done (cheap fix)
        │   └─ Storage doesn't help → Try logsumexp matmul
        │       ├─ Logsumexp helps → Done (expensive but works)
        │       └─ Logsumexp doesn't help → Problem is elsewhere
        │           → Investigate LR, init, normalization
        │
        └─ States NOT collapsing → Problem is elsewhere
            → Investigate gradient flow, optimizer, etc.
```

---

## Appendix: Quick Reference

### Log-Space Identities

```python
log(sigmoid(x)) = -softplus(-x)
log(1 - sigmoid(x)) = -softplus(x)
log(a * b) = log(a) + log(b)
log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))  # logaddexp
log(softplus(x)) = log(log(1 + exp(x)))
```

### Stability Thresholds

```python
# Clamp log values to prevent overflow when converting to linear
log_h = log_h.clamp(min=-40, max=40)  # exp(-40) ≈ 0, exp(40) ≈ 2e17

# Add epsilon before log to prevent -inf
log_x = torch.log(x + 1e-8)

# Use logaddexp instead of log(exp(a) + exp(b))
result = torch.logaddexp(a, b)  # Numerically stable
```

### Key Insight Summary

> "Without the right implementation of these primitives, the basic SSD algorithm produces NaNs immediately during training (even with FP32)." - Tri Dao

The "right implementation" is:
1. **Store decay factors in log-space** (A = -softplus(A_raw))
2. **Use segment-sum instead of cumsum + subtract**
3. **Diagonal transitions** (simple) OR **logsumexp decomposition** (complex, see 8.6)

---

## Final Notes for Implementer

### What to Try First
1. Run the hidden state magnitude diagnostic (Phase 1)
2. Wait for Diagonal MHTR results
3. If needed, implement signed log storage (easiest)

### What We're Uncertain About
- Whether log-space is actually the bottleneck
- Whether the 2x overhead of logsumexp matmul is acceptable
- Whether diagonal structure alone fixes the gap

### What We Know Works
- Output selectivity (compete×silu) helps
- Full R matrices give more expressivity than diagonal
- tanh activation is important for bounding

### Contact
If the diagonal experiments show Mamba2-matching performance, skip all the log-space work - diagonal structure would be the answer.

If diagonal doesn't help and hidden states look fine, the problem is elsewhere (optimizer, init, architecture).

Only pursue log-space if we observe actual numerical underflow.

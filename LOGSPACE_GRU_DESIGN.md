# Log-Space Hybrid GRU with Mamba-Style Architecture

**Goal**: Create a deep recurrent GRU with Mamba's architectural principles:
- Log-space numerics for stability
- Residual connections between layers
- Layer normalization
- Target: ~1B parameters, 24-32 layers depth

---

## Architecture Overview

### Layer Structure (Mamba-inspired):

```
Input
  ↓
[LayerNorm → LogSpaceGRU → Residual] ← Layer 1
  ↓
[LayerNorm → LogSpaceGRU → Residual] ← Layer 2
  ↓
  ...
  ↓
[LayerNorm → LogSpaceGRU → Residual] ← Layer N
  ↓
Output Projection
```

Each block:
```python
def gru_block(x, h_prev):
    # Pre-normalization (like Mamba)
    x_norm = layer_norm(x)

    # GRU with log-space gating
    x_out, h_next = log_space_gru(x_norm, h_prev)

    # Residual connection
    x = x + x_out

    return x, h_next
```

---

## Log-Space GRU Cell Design

### Key Insight from minGRU:

**Normal space** (current HybridFusedGRU):
```python
r = sigmoid(i_r + h_r)
z = sigmoid(i_z + h_z)
n = tanh(i_n + r * h_n)
h_new = (1 - z) * h_prev + z * n
```

**Log space** (target):
```python
# Gates in log space
log_r = log_sigmoid(i_r + h_r)
log_z = -softplus(-gate_z)  # log(sigmoid(z))
log_one_minus_z = -softplus(gate_z)  # log(1 - sigmoid(z))

# Candidate in log space
log_n = log_tanh(i_n + exp(log_r) * h_n)

# Interpolation in log space using logaddexp
log_h_new = logaddexp(
    log_one_minus_z + log_h_prev,
    log_z + log_n
)

# Hidden state STAYS in log space between timesteps!
h_next = log_h_new

# Only exponentiate for output projection
output = exp(log_h_new)
```

### Why This Works:

1. **No multiplication chains** - additions instead
2. **No underflow** - logaddexp handles small values
3. **Stable gradients** - addition is stable in both directions
4. **Proven** - minGRU paper shows this works

---

## Triton Kernel Modifications

### Current HybridFusedGRU kernel (normal space):
```python
@triton.jit
def gru_cell_fused(...):
    r = tl.sigmoid(i_r + h_r)
    z = tl.sigmoid(i_z + h_z)
    n = tanh(i_n + r * h_n)
    h_new = (1.0 - z) * h_prev + z * n
```

### Log-space kernel (target):
```python
@triton.jit
def gru_cell_fused_log(...):
    # Compute gates
    gate_r = i_r + h_r
    gate_z = i_z + h_z

    # Log-space gates
    log_r = -tl.softplus(-gate_r)  # log(sigmoid)
    log_z = -tl.softplus(-gate_z)
    log_one_minus_z = -tl.softplus(gate_z)

    # Candidate (need to handle r * h_n in log space)
    # This is tricky: log(a * b) = log(a) + log(b)
    # But we have: n = tanh(i_n + r * h_n)
    #            = tanh(i_n + exp(log_r) * h_n)

    # Option 1: Keep r computation in normal space (hybrid)
    r = tl.exp(log_r)
    n_pre = i_n + r * h_n
    log_n = log_tanh(n_pre)

    # Interpolation in log space
    term1 = log_one_minus_z + log_h_prev
    term2 = log_z + log_n
    log_h_new = logaddexp(term1, term2)

    return log_h_new  # Keep in log space!
```

**Challenge**: Triton needs `logaddexp` implementation:
```python
def logaddexp(a, b):
    """Numerically stable log(exp(a) + exp(b))"""
    max_val = tl.maximum(a, b)
    return max_val + tl.log(tl.exp(a - max_val) + tl.exp(b - max_val))
```

---

## Parameter Calculation for 1B Model

### Current (700M params, depth=20):
- dim = 2048
- depth = 20
- expansion = 1.0
- dim_inner = 2048

Per layer:
- Input proj: 2048 × (3×2048) = 12.6M
- Hidden proj: 2048 × (3×2048) = 12.6M
- Total: ~25M per layer

20 layers: 500M + embeddings/head = ~700M total

### Target (1B params):

**Option 1: Keep depth=20, increase width**
- Available for GRU layers: 1000M - 200M (embeddings) = 800M
- Per layer: 800M / 20 = 40M
- Per layer = 2 × dim × 3×dim_inner (approx)
- 40M = 2 × dim × 3×(dim × expansion)

For expansion=1.0:
- 40M = 6 × dim²
- dim = sqrt(6.67M) = **2,582**

For expansion=1.5:
- 40M = 2 × dim × 3×(1.5×dim) = 9 × dim²
- dim = **2,108**

For expansion=2.0:
- 40M = 12 × dim²
- dim = **1,826**

**Option 2: Increase depth to 24 (Mamba-like)**
- Available: 800M
- Per layer: 800M / 24 = 33.3M

For expansion=1.5:
- 33.3M = 9 × dim²
- dim = **1,925**

For expansion=2.0:
- 33.3M = 12 × dim²
- dim = **1,667**

**Option 3: Increase depth to 32 (Full Mamba)**
- Available: 800M
- Per layer: 800M / 32 = 25M

For expansion=2.0:
- 25M = 12 × dim²
- dim = **1,443**

### Recommended Configuration:

**Depth=24, dim=1920, expansion=1.5** (closest to current performance expectations):
- Per layer: 9 × 1920² = 33.2M
- 24 layers: 24 × 33.2M = 796M
- Total with embeddings: ~1B params ✓
- Reasonable matmul sizes (1920×2880)
- Mamba-like depth

---

## Implementation Plan

### Phase 1: Core Log-Space Implementation

1. **Create `LogSpaceHybridGRU` class** (`mingru/logspace_hybrid_gru.py`)
   - Start by copying `HybridFusedGRU`
   - Modify to use log-space hidden states
   - Add helper functions: `log_sigmoid`, `log_tanh`, `logaddexp`

2. **Modify Triton kernel**
   - Implement log-space gating
   - Test numerics carefully (no NaN, no Inf)
   - Ensure backward pass works correctly

3. **Unit tests**
   - Compare to normal-space GRU on small examples
   - Verify gradient flow is stable
   - Test with extreme values (very small/large)

### Phase 2: Residual Connections + LayerNorm

1. **Update `minLM.py`**
   - Add `nn.LayerNorm` before each GRU layer
   - Add residual connection after each layer
   - Handle hidden state dimensions correctly

2. **Architecture changes**
   ```python
   class DeepGRULM(nn.Module):
       def __init__(self, ...):
           self.layers = nn.ModuleList([
               LogSpaceHybridGRU(...) for _ in range(depth)
           ])
           self.layer_norms = nn.ModuleList([
               nn.LayerNorm(dim) for _ in range(depth)
           ])

       def forward(self, x, prev_hiddens=None):
           hiddens = []
           for i, (ln, gru) in enumerate(zip(self.layer_norms, self.layers)):
               residual = x
               x = ln(x)
               x_out, h = gru(x, prev_hiddens[i] if prev_hiddens else None)
               x = residual + x_out  # Residual connection
               hiddens.append(h)
           return x, hiddens
   ```

### Phase 3: Training and Validation

1. **Small model test** (3 layers, dim=512)
   - Train for 10k steps
   - Verify gradients stay healthy
   - Compare loss to baseline

2. **Deep model test** (24 layers, dim=1920)
   - Train for 50k steps
   - Monitor gradient norms across all layers
   - Verify no gradient death

3. **Full 1B training**
   - Resume from checkpoint if possible
   - Target loss < 3.0 in first 100k steps
   - Compare to previous 700M run

---

## Testing Strategy

### Numerical Stability Tests:

```python
def test_logspace_stability():
    """Verify log-space operations don't produce NaN/Inf"""

    # Test extreme values
    very_small = torch.tensor([1e-10, 1e-20, 1e-30])
    very_large = torch.tensor([1e10, 1e20, 1e30])

    log_small = torch.log(very_small)
    log_large = torch.log(very_large)

    # Test logaddexp
    result = logaddexp(log_small, log_large)
    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))

    # Test with hidden states
    h_prev = log_small
    gates = torch.randn_like(h_prev)
    h_new = logspace_gru_cell(gates, h_prev)
    assert not torch.any(torch.isnan(h_new))
```

### Gradient Flow Tests:

```python
def test_deep_gradient_flow():
    """Verify gradients propagate through 24 layers"""

    model = DeepLogSpaceGRU(depth=24, dim=512)
    x = torch.randn(4, 64, 512, requires_grad=True)

    out, _ = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradients at different layers
    for i, layer in enumerate(model.layers):
        grad_norm = layer.input_projection.weight.grad.norm()
        print(f"Layer {i}: grad_norm = {grad_norm:.4f}")
        assert grad_norm > 1e-6  # Not vanished
        assert grad_norm < 1e3    # Not exploded
```

---

## Expected Outcomes

### If successful:

1. **Gradient norms stay stable** (0.1-1.0) across all 24 layers
2. **Loss decreases consistently** - no plateau at 4.7
3. **Reaches loss < 2.0** within reasonable time (compare to minGRU's 575 steps)
4. **No NaN/Inf** during training
5. **Faster convergence** than current 337k+ steps

### Success metrics:

- Loss < 3.0 in 50k steps (vs current stuck at 4.7)
- Gradient norm > 0.1 throughout training (vs current 0.06-0.07)
- Perplexity < 20 (vs current ~110)

---

## Risk Mitigation

### Potential Issues:

1. **Triton kernel complexity** - log-space operations are tricky
   - Mitigation: Start with PyTorch implementation, port to Triton later

2. **Backward pass numerics** - log-space gradients need care
   - Mitigation: Extensive unit testing with known gradients

3. **Memory overhead** - storing log-space states
   - Mitigation: Same memory as normal-space (just different values)

4. **Training instability** with residuals
   - Mitigation: Start with small learning rate, gradually increase

---

## Timeline Estimate

- **Phase 1** (Log-space core): 1-2 days
  - Implementation: 4-6 hours
  - Testing: 4-6 hours

- **Phase 2** (Residuals + LayerNorm): 1 day
  - Implementation: 2-3 hours
  - Testing: 2-3 hours

- **Phase 3** (Training validation): 2-3 days
  - Small model: 1 day
  - Deep model: 1 day
  - Full training: ongoing

**Total**: 4-6 days to validated deep log-space GRU

---

## References

- minGRU paper: [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201)
- Mamba paper: [Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- Current HybridFusedGRU: `mingru/hybrid_fused_gru.py`
- Log-space reference: `mingru/minGRU.py`

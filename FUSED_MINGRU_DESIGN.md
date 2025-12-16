# Fully Fused Triton MinGRU: Deep Architecture Analysis

## Problem: Kernel Launch Overhead

**Current HybridFusedGRU bottleneck:**
```
for t in range(512):  # Sequential timesteps
    gates_input = x[t] @ W_input       # Kernel 1
    gates_hidden = h[t-1] @ W_hidden   # Kernel 2
    h[t] = gru_cell(gates)             # Kernel 3 (Triton fused)
```

**Cost per layer:** 512 timesteps × 2 matmuls = 1,024+ kernel launches
**Total for depth=20:** 20,480 kernel launches → GPU stutter

---

## MinGRU Mathematical Foundation

```python
# Parallel formulation (what we want to fuse):
log_coeffs = -softplus(gate)            # log(1 - sigmoid(gate))
log_z = -softplus(-gate)                # log(sigmoid(gate))
log_tilde_h = log_g(hidden)             # where g(x) = x+0.5 if x>=0 else sigmoid(x)
log_values = log_z + log_tilde_h

# Associative scan (Heinsen method):
a_star = log_coeffs.cumsum(dim=1)
log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
log_h = a_star + log_h0_plus_b_star
```

**Key operations:**
1. Matmul: `x @ W_hidden, x @ W_gate`
2. Activations: `softplus`, `log_g`
3. **Scan: `cumsum`, `logcumsumexp`** ← Bottleneck!
4. Output projection

---

## Architecture Options (Ranked by Practicality)

### Option 1: Fused Matmul + Activations, PyTorch Scan (RECOMMENDED)

**Rationale:** PyTorch's `cumsum` and `logcumsumexp` are *insanely* optimized (CUB library, Thrust, decades of NVIDIA investment). Don't rewrite what works.

```
┌─────────────────────────────────────────────┐
│ TRITON KERNEL (Single Launch)              │
│  ┌───────────────────────────────────────┐ │
│  │ 1. Fused Dual Matmul                  │ │
│  │    - x @ W_hidden → hidden            │ │
│  │    - x @ W_gate   → gate              │ │
│  │    - Single tiled matmul kernel       │ │
│  │                                        │ │
│  │ 2. Fused Activations                  │ │
│  │    - log_coeffs = -softplus(gate)     │ │
│  │    - log_z = -softplus(-gate)         │ │
│  │    - log_tilde_h = log_g(hidden)      │ │
│  │    - log_values = log_z + log_tilde_h │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ PyTorch Ops (NVIDIA Optimized)             │
│  - a_star = cumsum(log_coeffs)             │
│  - log_out = logcumsumexp(log_values - a)  │
│  - log_h = a_star + log_out                │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ TRITON KERNEL (Single Launch)              │
│  - out = log_h.exp() @ W_out               │
│  - Fused exp + matmul                      │
└─────────────────────────────────────────────┘
```

**Kernel count per layer:** 3 total (vs 1,024!)
- 1 × Fused input processing
- 2 × PyTorch scan (cumsum + logcumsumexp)
- 1 × Fused output

**Pros:**
- 340× fewer kernel launches (1,024 → 3)
- PyTorch scan is O(log T) parallel depth
- Production-stable (PyTorch ops are battle-tested)
- Easy to implement (reuse existing Triton matmul patterns)

**Cons:**
- Still 3 GPU→CPU→GPU sync points (minor)
- Can't fuse scan into compute

---

### Option 2: Full Custom Triton with Manual Parallel Scan

**Rationale:** Maximum control, but **extremely complex** and unlikely to beat NVIDIA's CUB.

Parallel scan requires multi-pass algorithm:
```
Pass 1: Up-sweep (reduction tree)
  for d in range(log2(T)):
    parallel_for k in range(T):
      x[k] = op(x[k], x[k - 2^d])

Pass 2: Down-sweep (broadcast tree)
  ...similar recursive structure
```

**Problems:**
1. **Triton limitations:**
   - No recursion support
   - No cross-block synchronization (without atomics)
   - Limited shared memory (48KB L1, 192KB L2)

2. **Memory access patterns:**
   - Parallel scan requires O(log T) passes
   - Each pass: strided memory access (poor coalescing)
   - Bank conflicts in shared memory

3. **Numerical stability:**
   - `logcumsumexp` uses log-sum-exp trick internally
   - Custom implementation needs max reduction + stable sum
   - Hard to match PyTorch's numerical robustness

**Estimated speedup:** 0.8-1.2× (SLOWER than PyTorch, ironically)

**Verdict:** Not recommended unless you have GPU kernel wizardry team.

---

### Option 3: Chunked Fusion (PRAGMATIC MIDDLE GROUND)

**Idea:** Process in chunks of 64-128 timesteps, fuse within chunks.

```python
for chunk_start in range(0, T, CHUNK_SIZE):  # ~8 iterations for T=512
    chunk = x[chunk_start:chunk_start+CHUNK_SIZE]

    # TRITON: Fused matmul + activations for chunk
    hidden, gate, log_coeffs, log_values = fused_kernel(chunk)

    # PyTorch: Scan within chunk (small)
    log_h_chunk = scan(log_coeffs, log_values, prev_hidden)

    prev_hidden = log_h_chunk[:, -1]
```

**Kernel count:** 8 × 3 = 24 per layer (still 40× better than 1,024)

**Pros:**
- Reduces scan overhead (64 elements vs 512)
- Better memory locality
- Easier backward pass (smaller chunks)

**Cons:**
- Still sequential across chunks
- Not fully parallel

---

## Recommended Implementation Plan

### Phase 1: Fused Input Processing (IMMEDIATE)

**File:** `mingru/triton_mingru_fused.py`

```python
@triton.jit
def fused_mingru_input_kernel(
    # Input: [B, T, D]
    x_ptr,
    # Weights: [D, H], [D, H]
    W_hidden_ptr, W_gate_ptr,
    # Output: [B, T, H]
    hidden_out_ptr, gate_out_ptr,
    log_coeffs_ptr, log_values_ptr,
    # Dimensions
    B, T, D, H,
    stride_xB, stride_xT, stride_xD,
    stride_outB, stride_outT, stride_outH,
    # Tile sizes
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fuse:
    1. Dual matmul (hidden + gate projections)
    2. All activations (softplus, log_g)
    3. Prepare log-space values for scan

    Single kernel handles [B, BLOCK_T, H] tiles.
    """
    pid_b = tl.program_id(0)  # Batch
    pid_t = tl.program_id(1)  # Time tile
    pid_h = tl.program_id(2)  # Hidden tile

    # Load input tile: [BLOCK_T, D]
    t_start = pid_t * BLOCK_T
    offs_t = t_start + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    x_tile = tl.load(
        x_ptr + pid_b * stride_xB + offs_t[:, None] * stride_xT + offs_d[None, :] * stride_xD,
        mask=(offs_t[:, None] < T) & (offs_d[None, :] < D),
        other=0.0
    )

    # Load weight tiles: [D, H]
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    W_hidden_tile = tl.load(
        W_hidden_ptr + offs_d[:, None] * H + offs_h[None, :],
        mask=(offs_d[:, None] < D) & (offs_h[None, :] < H),
        other=0.0
    )
    W_gate_tile = tl.load(
        W_gate_ptr + offs_d[:, None] * H + offs_h[None, :],
        mask=(offs_d[:, None] < D) & (offs_h[None, :] < H),
        other=0.0
    )

    # Fused matmul: hidden = x @ W_hidden, gate = x @ W_gate
    hidden = tl.dot(x_tile, W_hidden_tile, allow_tf32=True)
    gate = tl.dot(x_tile, W_gate_tile, allow_tf32=True)

    # Fused activations (all inline, no memory writes!)
    # log_coeffs = -softplus(gate) = -log(1 + exp(gate))
    log_coeffs = -tl.log(1.0 + tl.exp(-tl.abs(gate))) - tl.maximum(gate, 0.0)

    # log_z = -softplus(-gate) = log(sigmoid(gate))
    log_z = -tl.log(1.0 + tl.exp(-tl.abs(gate))) + tl.minimum(gate, 0.0)

    # log_g(x) = log(x + 0.5) if x >= 0 else log(sigmoid(x))
    log_tilde_h = tl.where(
        hidden >= 0,
        tl.log(hidden + 0.5),  # Positive case
        -tl.log(1.0 + tl.exp(-hidden))  # Negative case (log sigmoid)
    )

    # log_values = log_z + log_tilde_h
    log_values = log_z + log_tilde_h

    # Store everything
    out_offset = pid_b * stride_outB + offs_t[:, None] * stride_outT + offs_h[None, :] * stride_outH
    tl.store(hidden_out_ptr + out_offset, hidden, mask=(offs_t[:, None] < T) & (offs_h[None, :] < H))
    tl.store(gate_out_ptr + out_offset, gate, mask=(offs_t[:, None] < T) & (offs_h[None, :] < H))
    tl.store(log_coeffs_ptr + out_offset, log_coeffs, mask=(offs_t[:, None] < T) & (offs_h[None, :] < H))
    tl.store(log_values_ptr + out_offset, log_values, mask=(offs_t[:, None] < T) & (offs_h[None, :] < H))


class FullyFusedMinGRU(nn.Module):
    """
    Production MinGRU with fused Triton kernels + PyTorch scans.

    Reduces from 1,024 kernel launches to 3 per layer.
    """
    def __init__(self, dim, expansion_factor=1.0):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        self.W_hidden = nn.Linear(dim, self.dim_inner, bias=False)
        self.W_gate = nn.Linear(dim, self.dim_inner, bias=False)
        self.W_out = nn.Linear(self.dim_inner, dim, bias=False)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        B, T, D = x.shape
        H = self.dim_inner

        # Allocate outputs
        hidden = torch.empty(B, T, H, device=x.device, dtype=x.dtype)
        gate = torch.empty(B, T, H, device=x.device, dtype=x.dtype)
        log_coeffs = torch.empty(B, T, H, device=x.device, dtype=x.dtype)
        log_values = torch.empty(B, T, H, device=x.device, dtype=x.dtype)

        # Launch fused kernel
        grid = lambda meta: (B, triton.cdiv(T, meta['BLOCK_T']), triton.cdiv(H, meta['BLOCK_H']))

        fused_mingru_input_kernel[grid](
            x, self.W_hidden.weight, self.W_gate.weight,
            hidden, gate, log_coeffs, log_values,
            B, T, D, H,
            x.stride(0), x.stride(1), x.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            BLOCK_T=64, BLOCK_D=128, BLOCK_H=128
        )

        # PyTorch scan (optimized!)
        if prev_hidden is not None:
            log_values = torch.cat([prev_hidden, log_values], dim=1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

        a_star = log_coeffs.cumsum(dim=1)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
        log_h = a_star + log_h0_plus_b_star
        log_h = log_h[:, -T:]

        # Fused exp + output projection (TODO: another Triton kernel)
        out = self.W_out(log_h.exp())

        if return_next_prev_hidden:
            return out, log_h[:, -1:]
        return out
```

---

## Performance Predictions

### Current (HybridFusedGRU):
- **Kernel launches:** 20,480 per forward pass
- **Throughput:** 105k tokens/sec
- **Bottleneck:** CPU-GPU sync overhead

### Phase 1 (Fused Input + PyTorch Scan):
- **Kernel launches:** 60 per forward pass (340× reduction!)
- **Estimated throughput:** 180-220k tokens/sec (1.7-2.1× speedup)
- **Rationale:**
  - Eliminates 99.7% of kernel launches
  - PyTorch scan is O(log T) parallel
  - Fused matmul+activations saves bandwidth

### Memory Efficiency:
- **Current:** B×T×H intermediate for each matmul
- **Fused:** Same (can't avoid - need scan inputs)
- **Not worse:** No memory regression

---

## Implementation Priority

**Week 1:** Implement Phase 1 (fused input kernel)
- Expected: 80-100% speedup
- Risk: Low (PyTorch scans are proven)

**Week 2:** Profile and optimize tile sizes
- Tune BLOCK_T, BLOCK_H for your GPU
- A100: Try BLOCK_T=64, BLOCK_H=128
- H100: Try BLOCK_T=128, BLOCK_H=256

**Week 3+:** (Optional) Fused output kernel
- Fuse exp() + output matmul
- Marginal gain (~5-10%) but cleaner

---

## Why NOT Full Custom Scan?

**Brutal honesty:** NVIDIA has spent *years* optimizing CUB/Thrust prefix scans. Their implementation:
- Uses warp-level primitives (`__shfl_down_sync`)
- Exploits SM-specific memory hierarchies
- Has hand-tuned bank conflict avoidance
- Handles edge cases (numerical stability, mixed precision)

**Your custom Triton scan:** You'd need to reimplement all of this in Triton, which:
- Doesn't expose warp intrinsics
- Has limited shared memory control
- No cross-block sync (except slow atomics)

**Expected result:** 6 months of work to match 80% of PyTorch performance. Not worth it.

---

## Backward Pass Considerations

The fused kernel needs custom autograd:

```python
class FusedMinGRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_hidden, W_gate):
        # Run fused kernel
        hidden, gate, log_coeffs, log_values = fused_kernel(x, W_hidden, W_gate)
        ctx.save_for_backward(x, W_hidden, W_gate, hidden, gate, log_coeffs)
        return log_coeffs, log_values

    @staticmethod
    def backward(ctx, grad_log_coeffs, grad_log_values):
        # Compute gradients (fused if possible)
        # This is complex - needs careful derivative chain
        ...
```

**Strategy:** Start with PyTorch autograd (let it handle), then fuse backward in Phase 2.

---

## Conclusion

**Recommended:** Option 1 (Fused Input + PyTorch Scan)
- 340× fewer kernel launches
- 1.7-2.1× speedup predicted
- Low risk, battle-tested components
- Implementable in 1 week

**Not Recommended:** Custom Triton scan
- Extreme complexity
- Unlikely to beat NVIDIA's primitives
- High risk, marginal gain

Should I implement the Phase 1 fused kernel?

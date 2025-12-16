# PersistentGRU Race Condition Bug Report

## Summary
PersistentGRU kernel fails with NaN at step 4 for chunk_size=2048 but works at chunk_size=256. Root cause: **missing synchronization barrier between parallel programs writing/reading shared ping-pong buffers**.

## The Bug

### Kernel Architecture
- Grid: `(B, ceil_div(H, BLOCK_H))` - launches multiple programs per batch
- Each program handles one (batch, hidden_tile) persistently across all T timesteps
- Programs write to shared ping-pong buffers (H_prev/H_next)
- **Problem**: All programs must read the ENTIRE buffer written by OTHER programs

### Race Condition Timeline

**File**: `/home/erikg/gruboros/mingru/persistent_gru.py`

**Timestep t=0:**
```python
# Line 188-189: All 16 programs (for H=2048) write their tiles
tl.store(write_ptr + b * stride_h_b + h_n_offs * stride_h_h, h_new, mask=h_n_mask)
```
- Program 0 writes h[0:128]
- Program 1 writes h[128:256]
- ...
- Program 15 writes h[1920:2048]

**Timestep t=1:**
```python
# Line 128-134: All programs try to read ENTIRE buffer
for k0 in range(0, H, BLOCK_K):  # Iterates over ALL of H!
    h_k = tl.load(read_ptr + b * stride_h_b + k_offs * stride_h_h, ...)
```

**THE RACE**: Program 0 may reach t=1 and try to read h[1920:2048] before Program 15 has finished writing it at t=0!

**No synchronization**: Triton persistent kernels don't automatically barrier between loop iterations when programs have cross-dependencies.

### Additional Issue: Document Reset Race (Lines 106-111)

```python
if have_mask:
    reset_flag = tl.load(mask_reset_ptr + t * B + b).to(tl.float32)
    h_prev = h_prev * (1.0 - reset_flag)
    # Write back reset h_prev so K-loop reads correct values
    tl.store(read_ptr + b * stride_h_b + h_n_offs * stride_h_h, h_prev, mask=h_n_mask)
```

This writes back to the SAME buffer being read from, creating a write→read dependency **within the same timestep** across programs!

## Why It Works at T=256 but Fails at T=2048

### T=256 (batch=48):
- More work per program (48 batches vs 16)
- Better SM utilization
- Programs naturally stay more synchronized
- Race condition exists but rarely triggers

### T=2048 (batch=16):
- Less work per program
- More timing variance between programs
- Program 0 can get far ahead of Program 15
- Race condition triggers consistently → NaN at step 4

## Reproduction

```bash
# WORKS (100 steps):
./train.persistent_1b_test.sh  # T=256, batch=48

# FAILS (NaN at step 4):
./train.persistent_1b_2k_fixed.sh  # T=2048, batch=16
```

## The Fix

### Option 1: Separate Kernel Launches (RECOMMENDED)

Don't use persistent-over-time. Launch separate kernels for each timestep:

```python
for t in range(T):
    # Launch kernel for timestep t
    gru_single_timestep_kernel[grid](...)
    # GPU automatically synchronizes between kernel launches
```

**Pros**: Simple, guaranteed correct, easy to debug
**Cons**: Kernel launch overhead (~5-10μs per launch)

### Option 2: Single-Program-Per-Batch Architecture

One program per batch handles entire H dimension:

```python
# Grid: (B,) instead of (B, H_tiles)
# Each program loops over H tiles sequentially
```

**Pros**: No cross-program dependencies, still persistent-T
**Cons**: Less parallelism, may be slower

### Option 3: Explicit Barriers (NOT POSSIBLE)

Triton doesn't support global memory barriers across programs in a grid. `tl.debug_barrier()` only works within a single program's shared memory.

## Recommended Fix: Option 1

Replace the persistent kernel with a non-persistent version that launches once per timestep. This is how most GRU kernels work (cuDNN, FlashRNN, etc.).

Example structure:
```python
def forward(self, x, ...):
    h = h0
    for t in range(T):
        h = gru_step_kernel(x[t], h, U, bias)  # Kernel launch
    return h
```

The per-timestep launch overhead is negligible compared to the compute cost for large models (1B parameters, depth=27).

## Evidence

### Failing Log (T=2048)
```
/home/erikg/gruboros/logs/persistent_gru_1b_2k_fixed_20251214_155635.log
Step 3: L=8.3476 G=3.3281
Step 4: L=7.9770 G=nan  ← NaN appears
Step 5: L=nan G=nan
```

### Working Log (T=256)
```
/home/erikg/gruboros/logs/persistent_gru_1b_20251214_160018.log
Step 22: L=7.4879 G=0.2812
Step 99: L=6.2036 G=0.4961  ← Stable through 100 steps
```

## Action Items

1. Implement non-persistent GRU kernel with per-timestep launches
2. Verify correctness against cuDNN at T=2048
3. Benchmark performance vs cuDNN
4. If persistent-T is critical for performance, implement Option 2 (single-program-per-batch)

## Notes

- This is a **systematic bug** in the kernel design, not a numerical stability issue
- The GRU equations are correct (verified against PyTorch)
- BF16 precision is fine (cuDNN works with bf16 at T=2048)
- The ping-pong buffer logic is correct, but missing inter-program synchronization

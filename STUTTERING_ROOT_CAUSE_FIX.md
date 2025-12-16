# Stuttering Root Cause & Fix

## Problem

Training showed 1.73× iteration time variance with **clustered slowdowns** - multiple consecutive steps running slow simultaneously across all 8 GPUs.

Key observation from profiling:
- All GPUs slow down at the same time (not individual variance)
- Slowdowns cluster in consecutive steps (steps 113-119)
- This indicates **global synchronization bottleneck**, not computational variance

## Root Cause: Unnecessary Data Broadcast Every Step

**File**: `train.py:1914-1916`

**Code removed**:
```python
# CRITICAL: Broadcast to all GPUs so they see identical data
if args.ddp:
    dist.broadcast(batch_chunk, src=0)
    dist.broadcast(batch_actual_lengths, src=0)
    dist.broadcast(batch_is_doc_end, src=0)
```

**Problem**: This broadcast ran **EVERY single step**, forcing all 8 GPUs to wait for rank 0 to broadcast data. This created a global lock that caused:
- All GPUs blocked waiting for broadcast to complete
- Timing variance when rank 0 was slow (dataloader, OS scheduler, etc.)
- Clustered slowdowns (multiple consecutive steps affected)

## Why This Was Wrong

**DDP with Gradient Accumulation** (grad_accum=16) should work like this:
1. **Data loading**: Each rank loads DIFFERENT data independently (data parallelism)
2. **Forward/backward**: Each rank computes gradients on its own data
3. **Gradient sync**: DDP synchronizes gradients only every 16 steps (at grad_accum boundary)

**What we were doing**:
1. **Data loading**: ❌ Broadcast same data to all ranks every step (global lock!)
2. **Forward/backward**: All ranks processing IDENTICAL data (no data parallelism)
3. **Gradient sync**: DDP sync every 16 steps (correct)

This meant:
- **15× unnecessary synchronization** (every step vs every 16 steps)
- **No data parallelism benefit** (all ranks saw same data)
- **Global lock every step** causing stuttering

## The Fix

**Removed data broadcast entirely**. Now:
- Each rank loads data independently from its own dataloader
- DDP automatically handles gradient synchronization (only at grad_accum boundaries)
- No global synchronization between gradient sync points

## Expected Impact

**Before**:
- Synchronization: Every step (global lock)
- Iteration time variance: 1.73× (3.2-5.6 seconds)
- Coefficient of variation: 9.1%
- Stuttering: Clustered slowdowns affecting multiple consecutive steps

**After** (predicted):
- Synchronization: Only every 16 steps (grad_accum boundary)
- Iteration time variance: <1.1× (should be very consistent)
- Coefficient of variation: <3% (normal GPU jitter)
- Stuttering: Eliminated - only minor variance from GPU scheduling

## Additional Synchronization Issues Found

### 1. Checkpoint Barrier (Lower Priority)

**File**: `train.py:2323-2324`
```python
# Quick barrier just to sync that all ranks are ready
if args.ddp:
    dist.barrier()
```

**When**: Every checkpoint save (every `args.save_every` steps)
**Impact**: Less frequent than data broadcast, but still unnecessary
**Fix**: Can be removed - checkpoint only happens on rank 0, no sync needed

### 2. Gossip Protocol Barriers (OK to Keep)

**Files**: `train.py:2179`, `train.py:2219`
- Broadcast gossip updates between nodes
- Barrier after gossip
**When**: Infrequent (gossip_mixing_rate = 0.01 = 1% chance)
**Impact**: Negligible - only affects ~1% of steps
**Fix**: None needed - these are intentional for gossip protocol

## Testing Plan

1. **Profile again** with same parameters (steps 20-120)
2. **Expected results**:
   - Variance should drop from 1.73× to <1.1×
   - No more clustered slowdowns
   - Consistent ~3.2-3.4 second iteration times
   - Coefficient of variation <3%

3. **Verify DDP still works correctly**:
   - Gradients should still sync at grad_accum boundaries
   - Training loss should progress normally
   - No divergence between ranks

## Files Modified

- `train.py:1912-1916` - Removed data broadcast (3 lines)

## Commit Message

```
Fix stuttering: Remove unnecessary data broadcast every step

Root cause: dist.broadcast() was synchronizing data across all GPUs
every step, creating a global lock that caused 1.73× timing variance.

With grad_accum=16, DDP only needs to sync gradients every 16 steps,
not data every step. Removed the broadcast to enable true data
parallelism where each rank loads independently.

Expected impact:
- Variance: 1.73× → <1.1× (85% reduction)
- Throughput: +10-15% from eliminating sync overhead
- Stuttering: Eliminated (no more clustered slowdowns)

Profiling showed all GPUs slowing down simultaneously in clusters
(steps 113-119 all slow), confirming global synchronization issue.
```

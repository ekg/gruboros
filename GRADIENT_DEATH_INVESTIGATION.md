# Gradient Death Investigation

## Problem Statement

After ~350k steps, training experiences **persistent gradient death**:
- Gradients decay from healthy 0.2-0.5 to dead 0.06-0.08
- Loss plateaus at ~4.7-4.8
- Happens with ALL optimizers (AdamW, SGD+momentum, plain SGD)
- Happens with ALL learning rates (0.001, 0.01, 0.1)
- Happens with/without gradient clipping
- Happens with/without fresh optimizer state

## Observations

### Training Performance
- **Throughput**: Excellent (~113k-120k tokens/sec)
- **Memory**: Stable (~6.8GB)
- **Loss**: Stuck at 4.5-4.8 plateau
- **Gradients**: Die within ~500 steps regardless of intervention

### What We've Tried
1. ✗ 2× learning rate (0.002)
2. ✗ 10× learning rate (0.01)
3. ✗ 100× learning rate (0.1)
4. ✗ Fresh optimizer state (reset Adam momentum/variance)
5. ✗ SGD with momentum (0.9)
6. ✗ Plain SGD (momentum=0.0)
7. ✗ Remove gradient clipping (grad_clip=0.0)

**Finding**: Gradient clipping at 1.0 made it worse, but removal didn't fix the core issue.

## Hypotheses to Investigate

### H1: Hidden State Corruption
**Theory**: Hidden states persist across batches and may accumulate numerical errors or get stuck in attractor states.

**Evidence**:
- RNNs maintain hidden state across chunks
- Hidden states reset at document boundaries, but maybe incorrectly?
- Code at train.py:2078-2152 manages hidden state carryover

**Test**: Add `--reset_hidden_always` flag to force blank hidden states every batch

**Code locations**:
- train.py:2078-2087 - Reset logic using `reset_next` mask
- train.py:2149-2152 - Hidden state detach and carryover
- train.py:2086-2087 - `reset_next` assigned from `is_doc_end`

**Potential bugs**:
- `reset_next` may not be initialized on first batch?
- Masked fill might not be working correctly?
- DDP might be desync'ing hidden states across GPUs?

### H2: Optimizer Momentum/Variance Accumulation
**Theory**: Even plain SGD might have issues due to Schedule-Free's internal state.

**Evidence**:
- Schedule-Free AdamW is the default
- Has `sf_beta` and `sf_beta2` parameters
- Maintains moving averages internally

**Test**: Use `--no-schedulefree` flag to use vanilla AdamW

**Code locations**:
- train.py:1605-1615 - Optimizer selection
- Default is Schedule-Free unless `--no-schedulefree` specified

### H3: Loss Landscape Trap
**Theory**: Model genuinely stuck in flat region. Loss ~4.7 might be a local minimum.

**Evidence**:
- Loss improvement minimal across all interventions
- Gradients small but not zero
- Model still "training" but not improving

**Test**:
- Start from earlier checkpoint (pre-plateau)
- Try completely different architecture
- Examine loss curve from beginning

### H4: Gradient Accumulation Bug
**Theory**: grad_accum=16 might be causing issues with hidden state management.

**Evidence**:
- Complex interaction between gradient accumulation and hidden state carryover
- Hidden states updated every chunk but gradients every 16 chunks

**Test**: Set `--grad_accum 1` (no accumulation)

**Code locations**:
- train.py:2163 - `should_optimize = accumulated_steps >= args.grad_accum`
- train.py:2114 - `accumulated_steps += 1`

### H5: DDP Gradient Synchronization Issue
**Theory**: Gradient accumulation + DDP + hidden states = potential desync

**Evidence**:
- DDP uses `no_sync()` during accumulation steps
- Hidden states are local per-GPU
- Gradients only sync every 16 steps

**Test**: Single GPU training (no DDP)

**Code locations**:
- train.py:2118-2131 - DDP sync logic
- train.py:2119-2125 - `no_sync()` context during accumulation

### H6: Numerical Precision / BF16 Issues
**Theory**: BF16 accumulation causing numerical instability

**Evidence**:
- Using `torch.autocast` with BF16
- Long training runs accumulate errors

**Test**: Train with FP32 (remove `--bf16`)

**Code locations**:
- train.py:2089 - `torch.autocast(..., enabled=args.bf16)`

### H7: Checkpoint Corruption
**Theory**: The checkpoint at step 337k is itself corrupted/pathological

**Evidence**:
- All interventions start from same checkpoint
- Gradient death happens consistently from that point

**Test**: Load checkpoint from step 300k, 250k, 200k

## Investigation Plan

### Phase 1: Quick Tests (< 1 hour each)
1. **Test H1a**: `--reset_hidden_always` - Force reset every batch
2. **Test H2**: `--no-schedulefree` - Vanilla AdamW
3. **Test H4**: `--grad_accum 1` - No gradient accumulation
4. **Test H6**: Remove `--bf16` - Full precision

### Phase 2: Deeper Investigation (1-2 hours each)
5. **Test H7**: Load earlier checkpoint (step 300k)
6. **Test H5**: Single GPU training
7. **Examine H3**: Plot full loss curve, look for earlier plateau

### Phase 3: Code Audit
8. Add debug logging to hidden state reset logic
9. Check `reset_next` initialization
10. Verify DDP hidden state consistency

## Next Steps

**Immediate Action**: Implement `--reset_hidden_always` flag and test if hidden states are the culprit.

**If hidden states ARE the problem**: Debug the reset logic, fix the bug
**If hidden states are NOT the problem**: Move to optimizer/precision tests

## Expected Outcomes

**Success**: Gradients stay healthy (0.2-0.5) for 1000+ steps
**Failure**: Gradients die to 0.06-0.08 within 500 steps

## Notes

- Current checkpoint: `/mnt/nvme2n1/erikg/minlms/20251115_160927_700m_gru_deep_f22912a/latest.pt`
- Model: 700M params, depth=20, dim=2048, HybridFusedGRU
- Data: The Pile, tiktoken p50k_base
- Batch: 114 per GPU × 8 GPUs × 16 grad_accum = ~7.5M tokens/update

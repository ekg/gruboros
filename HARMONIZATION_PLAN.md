# Test Harmonization Plan: Apples-to-Apples Comparison

## Problem

Previous experiments had inconsistent settings:
- **TBPTT**: GRU models had TBPTT enabled (hidden state carryover), giving them
  cross-chunk context. Mamba2 architecturally cannot do TBPTT (returns `None, None`).
- **Learning rate**: Mamba2 used 0.0006, GRU variants used 0.001
- **Weight decay**: Mamba2 used 0.1, GRU variants used 0.033

This makes loss curve comparisons unfair.

## Solution: Match Mamba2 Baseline

All new experiments should match the Mamba2 1B baseline settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| dim | 2048 | Standard |
| chunk_size | 512 | Context window |
| lr | 0.0006 | Conservative, stable |
| weight_decay | 0.1 | Mamba2 default |
| grad_clip | 1.0 | Prevent explosions |
| train_steps | 10000 | Quick comparison |
| bf16 | yes | Performance + stability |
| TBPTT | **disabled** | Fair comparison |
| tokenizer | p50k_base | 50,257 tokens |
| GPUs | 8 | DDP mode |

Depth varies by architecture to hit ~1B params.

## Models to Test

### Phase 1: ElmanSilu (Current)
- **Script**: `train.elman_silu_1b.sh`
- **Architecture**: `h_new = tanh(h_candidate) * silu(gate)`
- **Backend**: haste CUDA kernels (3x faster than cuDNN GRU)
- **Depth**: 32 (~1.01B params)
- **Status**: Ready to run with harmonized settings

### Phase 2: GRU Variants (Need --no-tbptt)
These scripts need `--no-tbptt` added for fair comparison:
- `train.cudnn_mult_gru_silu.sh` - GRU with SiLU gate
- `train.standard_gru.sh` - Stock cuDNN GRU
- `train.cudnn_mult_gru_*.sh` - Various Mult GRU configs

### Already Fair (No TBPTT architecturally)
- `train.mamba2_1b.sh` - SSM, no recurrent state
- `train.mamba2_ffn3_1b.sh` - Mamba2 + FFN

## Metrics

Compare on:
1. **Loss at 10k steps** - Primary metric
2. **Throughput (tok/s)** - Training speed
3. **Memory usage** - Batch size capacity
4. **Loss curve shape** - Learning dynamics

## Expected Outcomes

| Model | Expected Loss @10k | Throughput |
|-------|-------------------|------------|
| Mamba2 | ~3.2 (baseline) | ~350k tok/s |
| ElmanSilu | TBD | ~900k tok/s (3x faster) |
| Mult GRU (no TBPTT) | TBD | ~300k tok/s |

If ElmanSilu matches or beats Mamba2 at 3x the speed, it's a significant win.

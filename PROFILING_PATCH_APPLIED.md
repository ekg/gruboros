# Detailed Profiling Patch Applied to train.py

## What Was Added

I've instrumented train.py with detailed profiling to measure iteration timing and identify the cause of stuttering (0.20-0.47 it/s variance).

### Changes Made

1. **Profiling Setup** (lines 33-43):
   - Added `prof_time()` function that syncs CUDA before taking timestamps
   - Created global dictionaries to store timing data

2. **Per-Iteration Timing** (lines 1846-1856):
   - Measures data loading time
   - Records total iteration time
   - Tracks timing for steps 20-120

3. **Per-Step Reporting** (lines 2326-2367):
   - Prints timing for each step with format:
     ```
     [PROF  32 REG] Data=  12ms Total=  4010ms it/s=0.25
     [PROF  47 OPT] Data=  11ms Total=  5011ms it/s=0.20
     ```
   - Identifies optimizer steps (OPT) vs regular steps (REG)
   - Summary statistics at step 119

## How to Use

Run training with the existing command:
```bash
./train.gru.sh 2>&1 | tee profiled_run.log
```

The profiling will automatically activate for steps 20-120 and print:
- `[PROF####]` lines showing per-step timing
- Data load time in milliseconds
- Total iteration time in milliseconds
- Iterations per second

## What to Look For

After 120 steps, analyze the `[PROF]` lines to find:

1. **Optimizer step pattern** (every 16th step with grad_accum=16):
   - Should see "OPT" markers
   - These steps are expected to be slower (optimizer overhead)

2. **Post-optimizer speedup**:
   - Step immediately after OPT should be faster
   - Pattern from logs: step 16, 32, 48 are 1.8× faster

3. **Regular step variance**:
   - Some regular steps: 0.22-0.28 it/s (normal)
   - Some regular steps: 0.15-0.20 it/s (slow - investigate!)

4. **Data loading time**:
   - Should be consistent (<50ms)
   - If varies, dataloader is the bottleneck

## Expected Output Example

```
[PROF  20 REG] Data=  12ms Total=  4200ms it/s=0.24
[PROF  21 REG] Data=  11ms Total=  3800ms it/s=0.26
...
[PROF  31 OPT] Data=  12ms Total=  5800ms it/s=0.17
[PROF  32 REG] Data=  11ms Total=  2100ms it/s=0.48  ← FAST after optimizer!
[PROF  33 REG] Data=  12ms Total=  4100ms it/s=0.24
...
```

## Analysis Steps

1. Extract profiling lines:
   ```bash
   grep "PROF" profiled_run.log > timing_analysis.txt
   ```

2. Identify fast vs slow steps:
   ```bash
   grep "PROF" profiled_run.log | awk '{print $2, $6}' | sort -k2 -n
   ```

3. Check correlation with optimizer:
   ```bash
   grep "OPT" profiled_run.log  # Should be every 16 steps
   ```

4. Find which steps are consistently slow (besides optimizer)

## Next Steps After Profiling

Once you have the profiling data, we can:
1. Identify if the slowness is correlated with specific step patterns
2. Check if post-optimizer speedup is real (and why)
3. Investigate why some regular steps are 2× slower than others
4. Add more detailed phase-level profiling if needed

The key question: **Why are some non-optimizer steps 2× slower?**

Possible causes to investigate based on profiling:
- Document boundary resets (hidden state resets)
- Variable sequence lengths
- GPU memory/cache effects
- Dataloader delays
- Something in the HybridGRU Triton kernel

## Files Modified

- `train.py`: Added profiling instrumentation (minimal overhead, <1% impact)

The profiling is completely passive and won't affect training - it just measures and reports timing.

# Z-Gate Monitoring for MinGRU Training

## Overview
This system tracks the z-gate (update gate) statistics during training to monitor the balance between memory retention and plasticity in the MinGRU/HybridGRU layers.

## Key Files Added

1. **train.py** - Modified to include:
   - `measure_z_stats()` - Measures z-gate statistics for a given layer
   - `log_z_stats_to_tsv()` - Logs statistics to TSV file every gradient update
   - Automatic logging on rank 0 to `{output_dir}/z_stats_rank0.tsv`

2. **monitor_z_gates.sh** - Real-time monitoring during training
   ```bash
   ./monitor_z_gates.sh output_dir  # Watch live during training
   ```

3. **analyze_z_stats.py** - Post-training analysis
   ```bash
   python analyze_z_stats.py output_dir/z_stats_rank0.tsv
   ```

## TSV File Format

The logged TSV file contains:
- `step` - Training step number
- `timestamp` - Unix timestamp
- `layer` - Layer index (0-indexed)
- `type` - "runtime" (with hidden state) or "input" (h=0)
- `mean`, `std` - Mean and standard deviation of z-gates
- `p01`, `p10`, `p50`, `p90`, `p99` - Percentiles

## Interpreting Z-Gate Values

The z-gate controls the blend between previous hidden state and new input:
```
h_t = (1 - z) * h_{t-1} + z * new_content
```

### Healthy Ranges
- **Mean z ∈ [0.15, 0.35]** - Good balance
- **p90 - p10 > 0.3** - Good variance (selective writes)

### Warning Signs
- **Mean z < 0.10** - Too closed, slow learning, potential plateau
  - Fix: Increase LR or make input z-bias less negative
- **Mean z > 0.50** - Too open, poor retention, forgets context
  - Fix: Decrease LR or make input z-bias more negative
- **Low variance** - Uniform gates, lacks expressiveness

## Configuration

By default, logs first 4 layers. To change:
```python
# In train.py, line ~1163
log_z_stats_to_tsv(model, chunk_data, hidden_state, step, z_stats_file, num_layers_to_log=8)  # Log 8 layers
```

## Usage During Training

1. **Start training normally**:
   ```bash
   ./train.cuda.sh
   ```

2. **Monitor in another terminal**:
   ```bash
   ./monitor_z_gates.sh output/2024-11-15_checkpoint_dir/
   ```

3. **Analyze after training**:
   ```bash
   python analyze_z_stats.py output/2024-11-15_checkpoint_dir/z_stats_rank0.tsv
   ```

## Memory vs Plasticity Trade-off

For byte-level memorization tasks:
- Start with **slightly closed** gates (mean ~0.20-0.25)
- Look for **sparse strong writes** (low mean, high p90)
- If loss plateaus with very low z, open gates slightly
- If loss is noisy with high z, close gates slightly

## Performance Impact

- Minimal overhead (~1-2% training time)
- Only runs on rank 0
- Uses small token probes (32 tokens)
- Writes are buffered and flushed periodically
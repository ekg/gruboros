#!/usr/bin/env python3
"""
Analyze z-gate statistics from TSV files generated during training.
Usage: python analyze_z_stats.py z_stats_rank0.tsv
"""

import sys
import pandas as pd
import numpy as np

def analyze_z_stats(tsv_file):
    """Load and analyze z-gate statistics from TSV file."""
    
    # Read the TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"Loaded {len(df)} rows from {tsv_file}")
    print(f"Steps covered: {df['step'].min()} to {df['step'].max()}")
    print(f"Layers tracked: {df['layer'].unique()}")
    print()
    
    # Group by layer and type to show latest stats
    latest_step = df['step'].max()
    latest_df = df[df['step'] == latest_step]
    
    print(f"=== Latest Z-gate statistics (step {latest_step}) ===")
    print()
    
    for layer in sorted(latest_df['layer'].unique()):
        layer_df = latest_df[latest_df['layer'] == layer]
        
        # Runtime stats
        runtime = layer_df[layer_df['type'] == 'runtime'].iloc[0]
        input_only = layer_df[layer_df['type'] == 'input'].iloc[0]
        
        print(f"Layer {layer}:")
        print(f"  Runtime:   mean={runtime['mean']:.3f}, std={runtime['std']:.3f}, " 
              f"[p10={runtime['p10']:.3f}, p50={runtime['p50']:.3f}, p90={runtime['p90']:.3f}]")
        print(f"  Input-only: mean={input_only['mean']:.3f}, std={input_only['std']:.3f}, "
              f"[p10={input_only['p10']:.3f}, p50={input_only['p50']:.3f}, p90={input_only['p90']:.3f}]")
        print()
    
    # Analyze trends over time for layer 0
    print("=== Layer 0 trends over training ===")
    layer0_runtime = df[(df['layer'] == 0) & (df['type'] == 'runtime')]
    
    if len(layer0_runtime) > 1:
        # Sample evenly across training
        n_samples = min(10, len(layer0_runtime))
        sample_indices = np.linspace(0, len(layer0_runtime)-1, n_samples, dtype=int)
        samples = layer0_runtime.iloc[sample_indices]
        
        print(f"{'Step':>8} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
        print("-" * 48)
        for _, row in samples.iterrows():
            print(f"{row['step']:8.0f} {row['mean']:8.3f} {row['p10']:8.3f} {row['p50']:8.3f} {row['p90']:8.3f}")
    
    print()
    
    # Check for concerning patterns
    print("=== Health Check ===")
    latest_l0_runtime = latest_df[(latest_df['layer'] == 0) & (latest_df['type'] == 'runtime')].iloc[0]
    mean_z = latest_l0_runtime['mean']
    
    if mean_z < 0.10:
        print(f"⚠️  Layer 0 z-gate is VERY CLOSED (mean={mean_z:.3f} < 0.10)")
        print("   → Model may be stuck in carry-dominant mode")
        print("   → Consider: increasing LR or making input z-bias less negative")
    elif mean_z < 0.15:
        print(f"⚠️  Layer 0 z-gate is somewhat closed (mean={mean_z:.3f} < 0.15)")
        print("   → Monitor for plateau in loss")
    elif mean_z > 0.5:
        print(f"⚠️  Layer 0 z-gate is VERY OPEN (mean={mean_z:.3f} > 0.5)")
        print("   → Model may forget long-range context")
        print("   → Consider: decreasing LR or making input z-bias more negative")
    elif mean_z > 0.35:
        print(f"ℹ️  Layer 0 z-gate is somewhat open (mean={mean_z:.3f} > 0.35)")
        print("   → Good for fast learning but watch for instability")
    else:
        print(f"✓  Layer 0 z-gate looks healthy (mean={mean_z:.3f} in [0.15, 0.35])")
        print("   → Good balance of plasticity and retention")
    
    # Check spread
    p90 = latest_l0_runtime['p90']
    p10 = latest_l0_runtime['p10']
    spread = p90 - p10
    
    if spread > 0.5:
        print(f"✓  Good variance in z-gate (p90-p10={spread:.3f})")
        print("   → Model has selective write patterns")
    elif spread < 0.2:
        print(f"⚠️  Low variance in z-gate (p90-p10={spread:.3f})")
        print("   → Gates are too uniform, may lack expressiveness")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_z_stats.py z_stats_rank0.tsv")
        sys.exit(1)
    
    try:
        analyze_z_stats(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
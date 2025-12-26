#!/usr/bin/env python3
"""
December 26, 2025 - Architecture Comparison at ~1.3B scale
Comparing: Triple R, Mamba2, LLaMA Transformer, Low-rank R

All runs: 300 steps, batch_size=16, chunk_size=512, ~65k tok/step
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Colorblind-friendly palette
COLORS = {
    'triple_r': '#D55E00',    # Orange-red
    'mamba2': '#0072B2',      # Blue
    'llama': '#009E73',       # Green
    'lowrank': '#CC79A7',     # Purple/pink
}


def parse_log(log_file):
    """Parse training log for step/loss pairs."""
    steps, losses = [], []
    try:
        with open(log_file) as f:
            for line in f:
                if line.startswith("Step"):
                    m = re.search(r'Step\s+(\d+):\s+L=([0-9.]+)', line)
                    if m:
                        steps.append(int(m.group(1)))
                        losses.append(float(m.group(2)))
    except FileNotFoundError:
        pass
    return steps, losses


def smooth(y, steps, window=20):
    """Moving average smoothing. Returns (smoothed_steps, smoothed_values)."""
    if len(y) < window:
        return np.array(steps), np.array(y)
    smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
    offset = window // 2
    smoothed_steps = np.array(steps[offset:offset+len(smoothed)])
    return smoothed_steps, smoothed


def avg_last_n(steps, losses, n=50):
    """Get average of last n steps."""
    if len(steps) < n:
        return None
    last_n = [l for s, l in zip(steps, losses) if s >= max(steps) - (n-1)]
    return sum(last_n) / len(last_n)


# Colorblind-friendly palette - extended
COLORS['baseline'] = '#E69F00'   # Yellow-orange

# Colorblind-friendly palette - more colors
COLORS['gru'] = '#56B4E9'      # Cyan for GRU
COLORS['selective'] = '#F0E442'  # Yellow for Selective Triple R

# Runs to compare
RUNS = [
    ('logs/selective_triple_r_1.33b_20251226.log', 'SelectiveTripleR (1.33B)', 'selective'),
    ('logs/triple_r_1.35b_20251226.log', 'ElmanTripleR (1.35B)', 'triple_r'),
    ('logs/triple_r_run6.log', 'ElmanTripleR run1 (1.28B)', 'triple_r'),
    ('logs/baseline_compete_silu_20251226.log', 'LeakyCompeteSilu (1.15B)', 'baseline'),
    ('logs/leaky_compete_silu_1.33b_20251226.log', 'LeakyCompeteSilu (1.33B)', 'baseline'),  # NEW - worse!
    ('logs/real_mamba2_20251226_165100.log', 'Mamba2 run1 (1.33B)', 'mamba2'),
    ('logs/real_mamba2_20251226_172754.log', 'Mamba2 run2 (1.33B)', 'mamba2'),
    ('logs/standard_gru_300_20251226.log', 'StandardGRU (1.28B)', 'gru'),
    ('logs/llama_1b_20251226_175907.log', 'LLaMA Transformer (1.34B)', 'llama'),
    ('logs/lowrank_r_20251226_140609.log', 'Low-rank R (1.18B)', 'lowrank'),
]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    print("\n" + "="*70)
    print("Architecture Comparison - Dec 26, 2025")
    print("="*70)
    print(f"\n{'Architecture':<30} {'avg50':<10} {'Final Step':<10}")
    print("-"*55)

    results = []

    for log_file, label, color_key in RUNS:
        color = COLORS[color_key]

        if not os.path.exists(log_file):
            print(f"{label:<30} {'N/A':<10} Log not found")
            continue

        steps, losses = parse_log(log_file)
        if not steps:
            print(f"{label:<30} {'N/A':<10} Empty log")
            continue

        a50 = avg_last_n(steps, losses, 50)
        max_step = max(steps)
        results.append((label, a50, max_step, color))

        # Light raw trace
        alpha = 0.3 if 'run2' in label.lower() else 0.15
        ax1.plot(steps, losses, alpha=alpha, color=color, linewidth=0.5)

        # Smoothed line - use solid for main runs, dashed for duplicates
        sm_steps, sm = smooth(losses, steps)
        linestyle = '--' if 'run2' in label.lower() else '-'
        linewidth = 1.5 if 'run2' in label.lower() else 2.5
        ax1.plot(sm_steps, sm, color=color, linewidth=linewidth,
                 linestyle=linestyle, label=label)

        print(f"{label:<30} {a50:.3f}     {max_step}")

    print("-"*55)

    # Left plot: Training curves
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss (nats)', fontsize=12)
    ax1.set_title('Training Loss Curves (300 steps)', fontsize=13, fontweight='bold')
    ax1.set_ylim(4.5, 8.5)
    ax1.set_xlim(0, 320)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Right plot: Bar chart of avg50
    valid = [(l, a, c) for l, a, _, c in results if a is not None]
    valid.sort(key=lambda x: x[1])  # Sort by loss

    labels = [r[0].replace(' (', '\n(') for r in valid]
    values = [r[1] for r in valid]
    colors = [r[2] for r in valid]

    bars = ax2.barh(range(len(valid)), values, color=colors)
    ax2.set_yticks(range(len(valid)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('Average Loss (last 50 steps)', fontsize=12)
    ax2.set_title('Final Performance (avg50)', fontsize=13, fontweight='bold')
    ax2.set_xlim(4.8, 6.0)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('~1.3B Parameter Model Comparison\n'
                 'The Pile, 512-token chunks, 8× A100, ~65k tok/step',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = '/tmp/dec26_model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    # Print ranking
    print("\n" + "="*50)
    print("Ranking (by avg50 loss, lower is better):")
    print("="*50)
    for rank, (label, avg, _, _) in enumerate(sorted(results, key=lambda x: x[1] or 999), 1):
        if avg:
            print(f"  {rank}. {label}: {avg:.3f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
December 27, 2025 - Extended Run Comparison at ~1.3B scale
Comparing 1000-step runs: Selective Triple R vs Mamba2
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Colorblind-friendly palette
COLORS = {
    'selective': '#D55E00',    # Orange-red for Selective Triple R
    'mamba2': '#0072B2',       # Blue for Mamba2
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


def smooth(y, steps, window=30):
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


# 1000-step runs to compare
RUNS_1000 = [
    ('logs/selective_triple_r_1000_20251227.log', 'Selective Triple R (1.33B)', 'selective'),
    ('logs/mamba2_1000_20251227_014136.log', 'Mamba2 (1.33B)', 'mamba2'),
]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    print("\n" + "="*70)
    print("1000-Step Extended Comparison - Dec 27, 2025")
    print("="*70)
    print(f"\n{'Architecture':<35} {'avg50':<10} {'Final Step':<10}")
    print("-"*60)

    results = []

    for log_file, label, color_key in RUNS_1000:
        color = COLORS[color_key]

        if not os.path.exists(log_file):
            print(f"{label:<35} {'N/A':<10} Log not found: {log_file}")
            continue

        steps, losses = parse_log(log_file)
        if not steps:
            print(f"{label:<35} {'N/A':<10} Empty log")
            continue

        a50 = avg_last_n(steps, losses, 50)
        max_step = max(steps)
        results.append((label, a50, max_step, color))

        # Light raw trace
        ax1.plot(steps, losses, alpha=0.15, color=color, linewidth=0.5)

        # Smoothed line
        sm_steps, sm = smooth(losses, steps)
        ax1.plot(sm_steps, sm, color=color, linewidth=2.5, label=f'{label}: avg50={a50:.3f}')

        print(f"{label:<35} {a50:.3f}     {max_step}")

    print("-"*60)

    # Left plot: Training curves
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss (nats)', fontsize=12)
    ax1.set_title('1000-Step Training Curves', fontsize=13, fontweight='bold')
    ax1.set_ylim(3.0, 8.5)
    ax1.set_xlim(0, 1050)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Add vertical line at 300 steps
    ax1.axvline(x=300, color='gray', linestyle='--', alpha=0.5, label='300 steps')
    ax1.text(310, 7.5, '300 steps\n(original comparison)', fontsize=9, color='gray')

    # Right plot: Bar chart of avg50
    valid = [(l, a, c) for l, a, _, c in results if a is not None]
    valid.sort(key=lambda x: x[1])  # Sort by loss

    labels = [r[0].replace(' (', '\n(') for r in valid]
    values = [r[1] for r in valid]
    colors = [r[2] for r in valid]

    bars = ax2.barh(range(len(valid)), values, color=colors, height=0.6)
    ax2.set_yticks(range(len(valid)))
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Average Loss (last 50 steps)', fontsize=12)
    ax2.set_title('Final Performance @ 1000 Steps', fontsize=13, fontweight='bold')
    ax2.set_xlim(3.5, 4.5)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=12, fontweight='bold')

    # Add delta annotation
    if len(values) == 2:
        delta = abs(values[0] - values[1])
        pct = delta / max(values) * 100
        ax2.text(4.0, 0.5, f'Delta: {delta:.3f} ({pct:.1f}%)',
                 fontsize=11, ha='center', style='italic')

    plt.suptitle('Mamba2 vs Selective Triple R @ 1000 Steps\n'
                 'The Pile, ~1.33B params, 8× A100, ~65k tok/step',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = '/tmp/1000_step_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for rank, (label, avg, steps, _) in enumerate(sorted(results, key=lambda x: x[1] or 999), 1):
        if avg:
            print(f"  {rank}. {label}: {avg:.3f}")

    if len(results) == 2:
        winner = min(results, key=lambda x: x[1])
        loser = max(results, key=lambda x: x[1])
        delta = loser[1] - winner[1]
        pct = delta / loser[1] * 100
        print(f"\n  Winner: {winner[0]}")
        print(f"  Advantage: {delta:.3f} ({pct:.1f}% lower loss)")


if __name__ == '__main__':
    main()

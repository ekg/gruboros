#!/usr/bin/env python3
"""
Leaky Elman RNN Training Comparison

Compares architectures with discretized continuous-time dynamics:
- Mamba2: Linear SSM baseline (parallelizable)
- ElmanLeaky: tanh + leaky integration + input-only silu gate (BEST RNN)
- ElmanLeakySelective: h+x output gate (worse - overdoing it)
- ElmanLeakySilu: silu + leaky integration (testing silu vs tanh)

All results from Dec 22, 2025 onwards (post loss-masking fix).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'blue': ('#0072B2', '#005a8d'),      # Mamba2
    'orange': ('#D55E00', '#a34700'),    # ElmanLeaky
    'yellow': ('#E69F00', '#b37d00'),    # ElmanLeakySelective
    'green': ('#009E73', '#007a59'),     # ElmanLeakySilu
    'purple': ('#CC79A7', '#a35f87'),    # Future use
    'cyan': ('#56B4E9', '#3d8fc4'),      # Future use
}


def smooth(y, steps, window=200):
    """Moving average smoothing. Returns (smoothed_steps, smoothed_values)."""
    if len(y) < window:
        return np.array(steps), np.array(y)
    smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
    offset = window // 2
    smoothed_steps = np.array(steps[offset:offset+len(smoothed)])
    return smoothed_steps, smoothed


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


def get_avg_loss(steps, losses, start=2000, end=3000):
    """Get average loss between start and end steps."""
    vals = [l for s, l in zip(steps, losses) if start <= s <= end]
    return sum(vals) / len(vals) if vals else None


# Post loss-masking-fix runs only (Dec 22, 2025 onwards)
# Format: (log_file, label, color_key, notes)
RUNS = [
    # === Mamba2 baseline (linear SSM) ===
    ('logs/mamba2_1b_matched_20251223_072401.log',
     'Mamba2 (linear SSM)',
     'blue',
     'Parallelizable via associative scan'),

    # === ElmanLeaky: BEST RNN (tanh + leaky + input-only gate) ===
    ('logs/elman_leaky_1b_20251223_171536.log',
     'ElmanLeaky (tanh)',
     'orange',
     'Best RNN: tanh candidate, input-only silu gate'),

    # === ElmanLeakySelective: h+x output gate (WORSE) ===
    ('logs/elman_leaky_selective_1b_20251224_131529.log',
     'ElmanLeakySelective (h+x gate)',
     'yellow',
     'Failed: h+x gate worse than input-only'),

    # === ElmanLeakySilu: silu instead of tanh ===
    ('logs/elman_leaky_silu_1b_new.log',
     'ElmanLeakySilu (silu)',
     'green',
     'Testing: silu candidate, input-only silu gate'),
]


def main():
    plt.figure(figsize=(14, 8))

    print("\n" + "="*60)
    print("Leaky Elman RNN Comparison (Post Loss-Masking Fix)")
    print("="*60)
    print(f"\n{'Architecture':<35} {'Avg Loss (2k-3k)':<18} {'Status'}")
    print("-"*70)

    for i, (log_file, label, color_key, notes) in enumerate(RUNS):
        light, dark = COLORS[color_key]

        steps, losses = parse_log(log_file)
        if not steps:
            print(f"{label:<35} {'N/A':<18} Log not found")
            continue

        # Light raw trace
        plt.plot(steps, losses, alpha=0.15, color=light, linewidth=0.5)

        # Heavy smoothed line
        sm_steps, sm = smooth(losses, steps)
        plt.plot(sm_steps, sm, color=dark, linewidth=2.5, label=label)

        # Calculate metrics
        avg_2k_3k = get_avg_loss(steps, losses)
        final = sm[-1] if len(sm) > 0 else losses[-1]
        final_step = sm_steps[-1] if len(sm_steps) > 0 else steps[-1]
        pct_complete = min(100, steps[-1] / 3000 * 100)

        # Status string
        if avg_2k_3k and steps[-1] >= 2500:
            status = f"Complete (final: {final:.2f})"
            display_label = f'{avg_2k_3k:.2f}'
            print(f"{label:<35} {avg_2k_3k:<18.3f} {status}")
        else:
            status = f"In progress ({pct_complete:.0f}%)"
            display_label = f'{final:.2f} ({pct_complete:.0f}%)'
            print(f"{label:<35} {'TBD':<18} {status}")

        # Annotate on plot
        plt.annotate(display_label, xy=(final_step, final),
                    xytext=(final_step + 100, final + 0.12 * (i - 1.5)),
                    fontsize=9, color=dark, fontweight='bold')

    print("-"*70)

    # Reference line
    plt.annotate('Random init: 10.82', xy=(0, 8), xytext=(100, 7.8),
                fontsize=10, color='#666')

    # Formatting
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Cross-Entropy Loss (nats)', fontsize=12)
    plt.title('Leaky Elman RNN: Architecture Comparison\n'
              '~1B params, The Pile, 512-token chunks, 8× A100',
              fontsize=13, fontweight='bold')
    plt.ylim(3.5, 8.5)
    plt.xlim(0, 4500)
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.grid(True, alpha=0.3, color='#ccc')
    plt.legend(loc='upper right', fontsize=9, framealpha=0.95)
    plt.tight_layout()

    # Save
    output_path = '/tmp/leaky_elman_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()

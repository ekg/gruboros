#!/usr/bin/env python3
"""
Leaky Elman RNN Comparison

Compares architectures with discretized dynamics:
- Mamba2: Linear SSM baseline
- ElmanLeakySelective: h+x output gate (overdoing it?)
- LeakyElman: Input-only output gate (closer to Mamba2's C matrix)

All runs use log-space A parameterization: decay_rate = exp(-exp(A_log))
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

def smooth(y, steps, window=200):
    """Smoothing for 3k step runs. Returns (smoothed_steps, smoothed_values) only where full window exists."""
    if len(y) < window:
        return steps, y
    smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
    # Offset steps to align with smoothed values (centered window)
    offset = window // 2
    smoothed_steps = steps[offset:offset+len(smoothed)]
    return smoothed_steps, smoothed

def parse_log(log_file):
    steps, losses = [], []
    with open(log_file) as f:
        for line in f:
            if line.startswith("Step"):
                m = re.search(r'Step\s+(\d+):\s+L=([0-9.]+)', line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    return steps, losses

# Post loss-masking-fix runs only (Dec 22, 2025 onwards)
logs = [
    # === Mamba2 baseline ===
    ('logs/mamba2_1b_matched_20251223_072401.log', 'Mamba2 (linear SSM)', '#0072B2', '#005a8d'),

    # === ElmanLeaky: True discretized, NO output gate - BEST RNN! ===
    ('logs/elman_leaky_1b_20251223_171536.log', 'ElmanLeaky (no gate, 3.9!)', '#D55E00', '#a34700'),

    # === ElmanLeakySelective: h+x output gate (log-space A fix) ===
    ('logs/elman_leaky_selective_1b_20251224_131529.log', 'ElmanLeakySelective (h+x gate)', '#E69F00', '#b37d00'),

    # === LeakyElman: input-only output gate (like Mamba2 C matrix) ===
    ('logs/leaky_elman_new.log', 'LeakyElman (input-only gate)', '#009E73', '#007a59'),
]

def get_avg_2k_3k(steps, losses):
    """Get average loss between steps 2000-3000."""
    vals = [l for s, l in zip(steps, losses) if 2000 <= s <= 3000]
    return sum(vals) / len(vals) if vals else None

plt.figure(figsize=(14, 8))
print("\n=== Average Loss (steps 2000-3000) ===")
for i, (log, label, light, dark) in enumerate(logs):
    try:
        steps, losses = parse_log(log)
        if not steps: continue
        # Light raw trace
        plt.plot(steps, losses, alpha=0.15, color=light, linewidth=0.5)
        # Heavy smoothed line (only where full window exists)
        sm_steps, sm = smooth(losses, steps)
        plt.plot(sm_steps, sm, color=dark, linewidth=2.5, label=label)

        # Calculate avg 2k-3k
        avg_2k_3k = get_avg_2k_3k(steps, losses)
        if avg_2k_3k:
            print(f"{label}: {avg_2k_3k:.3f}")

        final = sm[-1] if len(sm) > 0 else losses[-1]
        final_step = sm_steps[-1] if len(sm_steps) > 0 else steps[-1]
        # Show avg 2k-3k in annotation if available, otherwise final
        if avg_2k_3k and steps[-1] >= 2500:
            display_val = avg_2k_3k
            display_label = f'{avg_2k_3k:.2f}'
        else:
            display_val = final
            pct = f' ({steps[-1]/3000*100:.0f}%)' if steps[-1] < 2999 else ''
            display_label = f'{final:.2f}{pct}'
        plt.annotate(display_label, xy=(final_step, final),
                    xytext=(final_step+100, final + 0.12*(i-2)),
                    fontsize=9, color=dark, fontweight='bold')
    except: pass

plt.annotate('Random init: 10.82', xy=(0, 8), xytext=(100, 7.8), fontsize=10, color='#666')
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Cross-Entropy Loss (nats)', fontsize=12)
plt.title('Leaky Elman RNN: Output Gate Comparison\n~1B params, The Pile, 512-token chunks, 8× A100', fontsize=13, fontweight='bold')
plt.ylim(3.5, 8.5)
plt.xlim(0, 4500)
plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')
plt.grid(True, alpha=0.3, color='#ccc')
plt.legend(loc='upper right', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('/tmp/leaky_elman_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: /tmp/leaky_elman_comparison.png")

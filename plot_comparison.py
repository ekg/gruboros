#!/usr/bin/env python3
"""
Output Selection in Recurrent Language Models

Compares Stock GRU (no output selection) vs architectures with output selection:
- Mamba2: C matrix provides input-dependent output selection
- Mult GRU: Multiplicative gate h' = h * σ(Wx·x + Wh·h) filters outputs

Key finding: Output selection mechanism accounts for ~0.8 nats improvement.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

def smooth(y, steps, window=500):
    """Very heavy smoothing. Returns (smoothed_steps, smoothed_values) only where full window exists."""
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

# Output Selection Comparison
logs = [
    # === No Output Selection (Baseline) ===
    ('logs/test_standard_gru_no_tbptt_10k_20251215_215912.log', 'Stock GRU (no output selection)', '#FF0000', '#CC0000'),

    # === With Output Selection ===
    ('logs/mamba2_1b_20251207_060701.log', 'Mamba2 (C matrix selection)', '#0072B2', '#005a8d'),
    ('logs/cudnn_mult_gru_1b_20251210_044618.log', 'Mult GRU (multiplicative gate)', '#009E73', '#007a59'),
    ('logs/cudnn_mult_gru_512_20251212_022323.log', 'Mult GRU 512 ctx', '#2E8B57', '#1a5c38'),

    # === Variants ===
    ('logs/mamba2_ffn3_1b_20251207_234937.log', 'Mamba2 + FFN3', '#56B4E9', '#3d9fd4'),
    ('logs/cudnn_conv_gru_1b_20251211_175832.log', 'Conv + Mult GRU', '#4B0082', '#2d004d'),
    ('logs/cudnn_ffn3_gru_1b_20251211_212558.log', 'FFN3 Gate GRU', '#8B0000', '#5c0000'),

    # === Ablations (output selection variants) ===
    ('logs/input_only_gate_1b_20251209_035046.log', 'Input-only gate', '#E69F00', '#cc8a00'),
    ('logs/glu_gate_1b_20251209_160229.log', 'GLU gate', '#D55E00', '#b34d00'),
]

plt.figure(figsize=(14, 8))
for i, (log, label, light, dark) in enumerate(logs):
    try:
        steps, losses = parse_log(log)
        if not steps: continue
        # Light raw trace
        plt.plot(steps, losses, alpha=0.15, color=light, linewidth=0.5)
        # Heavy smoothed line (only where full window exists)
        sm_steps, sm = smooth(losses, steps)
        plt.plot(sm_steps, sm, color=dark, linewidth=2.5, label=label)
        final = sm[-1] if len(sm) > 0 else losses[-1]
        final_step = sm_steps[-1] if len(sm_steps) > 0 else steps[-1]
        pct = f' ({steps[-1]/10000*100:.0f}%)' if steps[-1] < 9999 else ''
        plt.annotate(f'{final:.2f}{pct}', xy=(final_step, final),
                    xytext=(final_step+100, final + 0.12*(i-2)),
                    fontsize=9, color=dark, fontweight='bold')
    except: pass

plt.annotate('Random init: 10.82', xy=(0, 8), xytext=(100, 7.8), fontsize=10, color='#666')
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Cross-Entropy Loss (nats)', fontsize=12)
plt.title('Output Selection in Recurrent LMs: Stock GRU vs Selective Architectures\n~1B params, The Pile, 512-token chunks, 8× A100', fontsize=13, fontweight='bold')
plt.ylim(2.5, 8.5)
plt.xlim(0, 10500)
plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')
plt.grid(True, alpha=0.3, color='#ccc')
plt.legend(loc='upper right', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('/tmp/output_selection_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: /tmp/output_selection_comparison.png")

#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

def smooth(y, window=50):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

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

logs = [
    ('logs/cudnn_ema_1b_10k_20251203_051425.log', 'Baseline (depth=23, no FF)', '#e74c3c', '#c0392b'),
    ('logs/cudnn_ema_ff4_20251203_192707.log', 'FF4 (depth=16, ff_mult=4)', '#3498db', '#2980b9'),
    ('logs/cudnn_ema_multiscale_20251204_140412.log', 'Multi-Scale EMA (3 timescales)', '#2ecc71', '#27ae60'),
    ('logs/cudnn_ssm_ff4_20251205_122750.log', 'Selective SSM FF4 (depth=14)', '#9b59b6', '#8e44ad'),
    ('logs/mamba2_1b_20251207_060701.log', 'Mamba2 SSD 1B (depth=35)', '#1abc9c', '#16a085'),
    ('logs/mamba2_ffn3_1b_20251207_234937.log', 'Mamba2+FFN3 (depth=15)', '#e67e22', '#d35400'),
    ('logs/cudnn_mult_gru_1b_20251208_170520.log', 'Mult GRU (depth=27)', '#f39c12', '#e67e22'),
    ('logs/cudnn_bilinear_gru_1b_20251208_212343.log', 'Bilinear GRU (depth=22)', '#c0392b', '#922b21'),
    ('logs/cudnn_plain_gru_1b_20251208_225720.log', 'Plain GRU (depth=36)', '#34495e', '#2c3e50'),
]

plt.figure(figsize=(14, 8))
for i, (log, label, light, dark) in enumerate(logs):
    try:
        steps, losses = parse_log(log)
        if not steps: continue
        plt.plot(steps, losses, alpha=0.2, color=light, linewidth=0.8)
        sm = smooth(losses)
        sm_steps = steps[len(steps)-len(sm):]
        plt.plot(sm_steps, sm, color=dark, linewidth=2.5, label=label)
        final = sm[-1] if sm.size else losses[-1]
        pct = f' ({steps[-1]/10000*100:.0f}%)' if steps[-1] < 9999 else ''
        plt.annotate(f'{final:.2f}{pct}', xy=(sm_steps[-1], final),
                    xytext=(sm_steps[-1]+100, final + 0.15*(i-1)),
                    fontsize=10, color=dark, fontweight='bold')
    except: pass

plt.annotate('Start: 10.82', xy=(0, 8), xytext=(100, 7.6), fontsize=11, color='#666')
plt.xlabel('Step'); plt.ylabel('Loss')
plt.title('1B Model Training Comparison: cuDNN GRU variants vs Mamba SSM\nPyTorch DDP, 8 GPUs')
plt.ylim(0, 8); plt.xlim(0, 10000)
plt.grid(True, alpha=0.3); plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/tmp/cudnn_ema_comparison.png', dpi=100, bbox_inches='tight')
print("Saved: /tmp/cudnn_ema_comparison.png")

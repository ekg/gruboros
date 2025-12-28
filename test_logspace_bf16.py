"""
Test log-space implementation with bfloat16 - more sensitive to numerical issues.
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from mingru.elman_ladder.log_storage_diagonal import LogStorageDiagonal, HASTE_AVAILABLE

print("=" * 70)
print("LOG-SPACE BFLOAT16 STABILITY TEST")
print("=" * 70)

device = 'cuda'
dtype = torch.bfloat16

dim = 512
batch = 4
n_groups = 32

# Create model in bfloat16
model = LogStorageDiagonal(
    dim=dim,
    expansion=1.0,
    n_groups=n_groups,
    delta_init=-2.0
).to(device).to(dtype)

print(f"Model dtype: {next(model.parameters()).dtype}")
print()

# Test increasing sequence lengths
print("Testing sequence lengths:")
print("-" * 50)

for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    try:
        with torch.no_grad():
            output, h_final = model(x)

        has_nan = torch.isnan(output).any().item() or torch.isnan(h_final).any().item()
        has_inf = torch.isinf(output).any().item() or torch.isinf(h_final).any().item()
        h_mean = h_final.abs().float().mean().item()
        out_mean = output.abs().float().mean().item()

        if has_nan or has_inf:
            print(f"seq_len={seq_len:5d}: FAIL (NaN={has_nan}, Inf={has_inf})")
        else:
            print(f"seq_len={seq_len:5d}: OK, h_mean={h_mean:.6f}, out_mean={out_mean:.6f}")
    except Exception as e:
        print(f"seq_len={seq_len:5d}: ERROR - {e}")

print()

# Training test with bfloat16
print("=" * 70)
print("TRAINING WITH BFLOAT16")
print("=" * 70)

model = LogStorageDiagonal(
    dim=dim,
    expansion=1.0,
    n_groups=n_groups,
    delta_init=-2.0
).to(device).to(dtype)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

seq_len = 1024
losses = []
nan_steps = 0

for step in range(50):
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    target = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    output, h_final = model(x)

    if torch.isnan(output).any() or torch.isnan(h_final).any():
        nan_steps += 1
        continue

    loss = F.mse_loss(output.float(), target.float())

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_nan_grad = any(torch.isnan(p.grad).any() if p.grad is not None else False for p in model.parameters())
    if has_nan_grad:
        nan_steps += 1
        continue

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.6f}")

print()
print(f"Completed {len(losses)}/50 steps")
print(f"NaN steps: {nan_steps}")
if losses:
    print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")

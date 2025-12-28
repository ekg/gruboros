"""
Test log-space implementation for training stability and accuracy at depth.

Compare training loss and numerical stability between:
1. Log-space output (current implementation)
2. Check for NaN/Inf issues
3. Training dynamics over multiple steps
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from mingru.elman_ladder.log_storage_diagonal import LogStorageDiagonal, HASTE_AVAILABLE

print("=" * 70)
print("LOG-SPACE TRAINING STABILITY TEST")
print("=" * 70)
print(f"HASTE available: {HASTE_AVAILABLE}")
print()

device = 'cuda'
dtype = torch.float32  # Use float32 for stability analysis

# Model config
dim = 256
batch = 4
n_groups = 16
seq_len = 512  # Longer sequence to test depth

# Create model
model = LogStorageDiagonal(
    dim=dim,
    expansion=1.0,
    n_groups=n_groups,
    delta_init=-2.0  # Small delta for long memory
).to(device).to(dtype)

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Config: dim={dim}, seq_len={seq_len}, batch={batch}")
print()

# Training loop
losses = []
nan_count = 0
inf_count = 0

for step in range(100):
    # Random input
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    # Random target (simplified: predict shifted input)
    target = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    # Forward
    output, h_final = model(x)

    # Check for numerical issues
    if torch.isnan(output).any():
        nan_count += 1
        print(f"Step {step}: NaN in output!")
        continue
    if torch.isinf(output).any():
        inf_count += 1
        print(f"Step {step}: Inf in output!")
        continue

    # Loss
    loss = F.mse_loss(output, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradient norms
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # Check for NaN gradients
    has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    if has_nan_grad:
        nan_count += 1
        print(f"Step {step}: NaN in gradients!")
        continue

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    losses.append(loss.item())

    if step % 20 == 0:
        print(f"Step {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.4f}")

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Completed {len(losses)} steps successfully")
print(f"NaN occurrences: {nan_count}")
print(f"Inf occurrences: {inf_count}")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Loss improved: {losses[-1] < losses[0]}")
print()

# Test longer sequences
print("=" * 70)
print("TESTING LONGER SEQUENCES")
print("=" * 70)

for test_len in [256, 512, 1024, 2048]:
    x = torch.randn(2, test_len, dim, device=device, dtype=dtype)

    with torch.no_grad():
        output, h_final = model(x)

    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    output_mean = output.abs().mean().item()
    output_max = output.abs().max().item()
    h_mean = h_final.abs().mean().item()

    status = "OK" if not has_nan and not has_inf else "FAIL"
    print(f"seq_len={test_len:4d}: {status}, output_mean={output_mean:.6f}, output_max={output_max:.6f}, h_final_mean={h_mean:.6f}")

print()
print("=" * 70)
print("NUMERICAL PRECISION TEST")
print("=" * 70)

# Test with very long sequence to check for underflow/overflow
x_long = torch.randn(1, 4096, dim, device=device, dtype=dtype)
with torch.no_grad():
    output_long, h_long = model(x_long)

# Check the hidden state at depth
print(f"Sequence length 4096:")
print(f"  h_final abs mean: {h_long.abs().mean().item():.6e}")
print(f"  h_final abs min:  {h_long.abs().min().item():.6e}")
print(f"  h_final abs max:  {h_long.abs().max().item():.6e}")
print(f"  output abs mean:  {output_long.abs().mean().item():.6e}")
print(f"  Has NaN: {torch.isnan(h_long).any().item()}")
print(f"  Has Inf: {torch.isinf(h_long).any().item()}")

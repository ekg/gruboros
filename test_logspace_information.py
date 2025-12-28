"""
Test if log-space hidden state maintains information at depth.

Key test: Can the model distinguish different inputs at early positions
when measuring at the final hidden state?
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn.functional as F
from mingru.elman_ladder.log_storage_diagonal import LogStorageDiagonal

print("=" * 70)
print("INFORMATION PRESERVATION AT DEPTH TEST")
print("=" * 70)

device = 'cuda'
dtype = torch.float32

dim = 256
n_groups = 16

model = LogStorageDiagonal(
    dim=dim,
    expansion=1.0,
    n_groups=n_groups,
    delta_init=-2.0  # Small delta = long memory
).to(device).to(dtype)

# Test: inject different signals at position 0, measure difference at final h
print("\nTest: Signal at position 0, measure at final hidden state")
print("-" * 50)

for seq_len in [64, 128, 256, 512, 1024, 2048]:
    # Two sequences that differ only at position 0
    x1 = torch.randn(1, seq_len, dim, device=device, dtype=dtype) * 0.1
    x2 = x1.clone()
    x2[:, 0, :] = x1[:, 0, :] + 1.0  # Add signal at position 0

    with torch.no_grad():
        _, h1 = model(x1)
        _, h2 = model(x2)

    # Measure difference in final hidden state
    h_diff = (h2 - h1).abs().mean().item()
    h1_norm = h1.abs().mean().item()
    relative_diff = h_diff / (h1_norm + 1e-10)

    print(f"seq_len={seq_len:4d}: h_diff={h_diff:.6e}, relative={relative_diff:.6e}")

print()
print("If relative difference > 0, information from position 0 reaches final h.")
print()

# Test: correlation between early input and final output
print("=" * 70)
print("INPUT-OUTPUT CORRELATION TEST")
print("=" * 70)
print()

for seq_len in [64, 256, 1024]:
    # Generate many random inputs
    n_samples = 100
    x = torch.randn(n_samples, seq_len, dim, device=device, dtype=dtype)

    with torch.no_grad():
        output, h_final = model(x)

    # Correlation between input at position 0 and output at last position
    x_first = x[:, 0, :].flatten()
    out_last = output[:, -1, :].flatten()

    # Simple correlation
    x_mean = x_first.mean()
    out_mean = out_last.mean()
    x_std = x_first.std()
    out_std = out_last.std()
    corr = ((x_first - x_mean) * (out_last - out_mean)).mean() / (x_std * out_std + 1e-10)

    print(f"seq_len={seq_len:4d}: corr(x[0], out[-1])={corr.item():.6f}")

print()
print("Non-zero correlation indicates information flows through the sequence.")
print()

# Test: hidden state magnitude over time
print("=" * 70)
print("HIDDEN STATE MAGNITUDE OVER TIME")
print("=" * 70)
print()

import haste_pytorch_lib

x = torch.randn(1, 512, dim, device=device, dtype=dtype)
log_h0 = torch.zeros(1, dim, device=device, dtype=dtype)
sign_h0 = torch.ones(1, dim, device=device, dtype=dtype)

# Get model weights
W_x = model.cell.W_x.data
r_h = model.cell.r_h.data
W_delta = model.cell.W_delta.data
W_out = model.cell.W_out.data
b = model.cell.b.data
b_delta = model.cell.b_delta.data

# Project input
x_proj = model.in_proj(x).permute(1, 0, 2).contiguous()

result = haste_pytorch_lib.log_storage_diagonal_forward(
    False, x_proj, log_h0, sign_h0, W_x, r_h, W_delta, W_out, b, b_delta, n_groups
)

log_h, sign_h, output = result[0], result[1], result[2]

# log_h has shape [T+1, B, D], track mean log|h| over time
log_h_means = log_h[:, 0, :].mean(dim=-1).cpu().numpy()  # [T+1]
h_means = (sign_h[:, 0, :] * torch.exp(log_h[:, 0, :])).abs().mean(dim=-1).cpu().numpy()

print("Time step | log|h| mean | |h| mean")
print("-" * 40)
for t in [0, 10, 50, 100, 200, 300, 400, 500]:
    if t < len(log_h_means):
        print(f"t={t:4d}     | {log_h_means[t]:11.4f} | {h_means[t]:.6e}")

print()
print("Log|h| should stay bounded (not go to -inf).")
print("|h| may decay but shouldn't underflow to 0.")

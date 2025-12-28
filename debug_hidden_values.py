"""
Debug hidden state values and magnitudes through stacked layers.
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn as nn
from mingru.elman_ladder.log_storage_diagonal import LogStorageDiagonal

device = 'cuda'
dtype = torch.float32

dim = 256
seq_len = 64
batch = 2
n_groups = 16
n_layers = 4

print("=" * 70)
print("HIDDEN STATE VALUE ANALYSIS")
print("=" * 70)

layers = nn.ModuleList([
    LogStorageDiagonal(dim=dim, expansion=1.0, n_groups=n_groups, delta_init=-2.0)
    for _ in range(n_layers)
]).to(device).to(dtype)

x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

print(f"\nInput: mean={x.abs().mean().item():.6f}, std={x.std().item():.6f}")
print()

# Forward through each layer, tracking h values
for i, layer in enumerate(layers):
    # Get internal values
    x_proj = layer.in_proj(x)
    x_rnn = x_proj.permute(1, 0, 2).contiguous()

    # Run cell and get h
    h_all, selective_out = layer.cell(x_rnn)
    h_final = h_all[-1]

    # Output
    selective_out_transposed = selective_out.permute(1, 0, 2).contiguous()
    output = layer.out_proj(selective_out_transposed)

    print(f"Layer {i}:")
    print(f"  x_proj mean: {x_proj.abs().mean().item():.6f}")
    print(f"  h_all mean: {h_all.abs().mean().item():.6f}")
    print(f"  h_final mean: {h_final.abs().mean().item():.6f}")
    print(f"  selective_out mean: {selective_out.abs().mean().item():.6f}")
    print(f"  output mean: {output.abs().mean().item():.6f}")
    print()

    x = output  # Pass to next layer

print("=" * 70)
print("LOG-SPACE VALUES ANALYSIS")
print("=" * 70)

# Fresh run with access to log_h
import haste_pytorch_lib

model = LogStorageDiagonal(dim=dim, expansion=1.0, n_groups=n_groups, delta_init=-2.0).to(device).to(dtype)
x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype) * 0.8  # Similar to input

# Get weights
W_x = model.cell.W_x.data
r_h = model.cell.r_h.data
W_delta = model.cell.W_delta.data
W_out = model.cell.W_out.data
b = model.cell.b.data
b_delta = model.cell.b_delta.data

# Project input
x_proj = model.in_proj(x).permute(1, 0, 2).contiguous()

# Initial state
log_h0 = torch.zeros(batch, model.d_inner, device=device, dtype=dtype)
sign_h0 = torch.ones(batch, model.d_inner, device=device, dtype=dtype)

result = haste_pytorch_lib.log_storage_diagonal_forward(
    False, x_proj, log_h0, sign_h0, W_x, r_h, W_delta, W_out, b, b_delta, n_groups
)

log_h, sign_h, output = result[0], result[1], result[2]

print(f"\nlog_h shape: {log_h.shape}")  # [T+1, B, D]
print(f"sign_h shape: {sign_h.shape}")

# Check log_h values at different timesteps
for t in [0, 10, 30, 63]:
    log_h_t = log_h[t]
    sign_h_t = sign_h[t]
    h_t = sign_h_t * torch.exp(log_h_t)

    print(f"\nTimestep {t}:")
    print(f"  log_h mean: {log_h_t.mean().item():.4f}, min: {log_h_t.min().item():.4f}, max: {log_h_t.max().item():.4f}")
    print(f"  sign_h mean: {sign_h_t.mean().item():.4f}")
    print(f"  h mean: {h_t.abs().mean().item():.6f}")
    print(f"  h min: {h_t.abs().min().item():.6f}")
    print(f"  h max: {h_t.abs().max().item():.6f}")

print()
print("=" * 70)
print("OUTPUT MAGNITUDE ANALYSIS")
print("=" * 70)

# What's the output magnitude?
print(f"\nOutput shape: {output.shape}")
print(f"Output mean: {output.abs().mean().item():.6f}")
print(f"Output std: {output.std().item():.6f}")

# What if we used h directly for output instead of log_h?
h = sign_h[1:] * torch.exp(log_h[1:])  # [T, B, D], exclude h0
print(f"\nh (real space) mean: {h.abs().mean().item():.6f}")
print(f"h (real space) std: {h.std().item():.6f}")

# Compute what W_out @ h would look like
w_out_h = torch.einsum('ij,tbj->tbi', W_out, h)
print(f"\nW_out @ h mean: {w_out_h.abs().mean().item():.6f}")
print(f"W_out @ h std: {w_out_h.std().item():.6f}")

# Compute what W_out @ log_h looks like
w_out_log_h = torch.einsum('ij,tbj->tbi', W_out, log_h[1:])
print(f"\nW_out @ log_h mean: {w_out_log_h.abs().mean().item():.6f}")
print(f"W_out @ log_h std: {w_out_log_h.std().item():.6f}")

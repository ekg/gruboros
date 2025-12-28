"""
Test log-space implementation at actual DEPTH (stacked layers).

Depth = number of stacked RNN layers, NOT sequence length.
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from mingru.elman_ladder.log_storage_diagonal import LogStorageDiagonal

print("=" * 70)
print("LOG-SPACE DEPTH TEST (STACKED LAYERS)")
print("=" * 70)

device = 'cuda'
dtype = torch.bfloat16

dim = 256
seq_len = 512  # Fixed sequence length
batch = 2
n_groups = 16

class StackedLogSpaceRNN(nn.Module):
    def __init__(self, dim, n_layers, n_groups=16):
        super().__init__()
        self.layers = nn.ModuleList([
            LogStorageDiagonal(dim=dim, expansion=1.0, n_groups=n_groups, delta_init=-2.0)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x

# Test increasing depth
print(f"\nConfig: dim={dim}, seq_len={seq_len}, batch={batch}")
print()
print("Testing depths:")
print("-" * 60)

for n_layers in [1, 2, 4, 8, 12, 16, 24, 32]:
    model = StackedLogSpaceRNN(dim, n_layers, n_groups).to(device).to(dtype)

    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    try:
        with torch.no_grad():
            output = model(x)

        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        out_mean = output.abs().float().mean().item()
        out_std = output.float().std().item()

        if has_nan or has_inf:
            print(f"depth={n_layers:2d} layers: FAIL (NaN={has_nan}, Inf={has_inf})")
        else:
            print(f"depth={n_layers:2d} layers: OK, out_mean={out_mean:.6f}, out_std={out_std:.6f}")
    except Exception as e:
        print(f"depth={n_layers:2d} layers: ERROR - {e}")

print()

# Training test at depth
print("=" * 70)
print("TRAINING AT DEPTH")
print("=" * 70)

for n_layers in [4, 12, 24]:
    print(f"\nTraining with {n_layers} layers:")
    print("-" * 40)

    model = StackedLogSpaceRNN(dim, n_layers, n_groups).to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    nan_count = 0

    for step in range(20):
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        target = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

        output = model(x)

        if torch.isnan(output).any():
            nan_count += 1
            continue

        loss = F.mse_loss(output.float(), target.float())

        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        has_nan_grad = any(
            torch.isnan(p.grad).any() if p.grad is not None else False
            for p in model.parameters()
        )
        if has_nan_grad:
            nan_count += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    if losses:
        print(f"  Completed: {len(losses)}/20 steps")
        print(f"  NaN steps: {nan_count}")
        print(f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    else:
        print(f"  FAILED - all steps had NaN")

print()
print("=" * 70)
print("GRADIENT FLOW AT DEPTH")
print("=" * 70)

for n_layers in [4, 12, 24]:
    model = StackedLogSpaceRNN(dim, n_layers, n_groups).to(device).to(torch.float32)

    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32, requires_grad=True)

    output = model(x)
    loss = output[:, -1, :].sum()  # Loss only at last position
    loss.backward()

    # Check gradient at input
    x_grad_mean = x.grad.abs().mean().item()
    x_grad_first = x.grad[:, 0, :].abs().mean().item()
    x_grad_last = x.grad[:, -1, :].abs().mean().item()

    # Check gradient at first layer vs last layer
    first_layer_grad = model.layers[0].cell.W_x.grad.abs().mean().item()
    last_layer_grad = model.layers[-1].cell.W_x.grad.abs().mean().item()

    print(f"\ndepth={n_layers:2d} layers:")
    print(f"  x_grad mean: {x_grad_mean:.6e}")
    print(f"  first_layer W_x grad: {first_layer_grad:.6e}")
    print(f"  last_layer W_x grad:  {last_layer_grad:.6e}")
    print(f"  ratio (first/last):   {first_layer_grad / (last_layer_grad + 1e-20):.6f}")

"""
Debug gradient flow through stacked log-space layers.
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
print("DEBUGGING GRADIENT FLOW THROUGH STACKED LAYERS")
print("=" * 70)

# Create layers
layers = nn.ModuleList([
    LogStorageDiagonal(dim=dim, expansion=1.0, n_groups=n_groups, delta_init=-2.0)
    for _ in range(n_layers)
]).to(device).to(dtype)

x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

# Forward through each layer, tracking activations
print("\nForward pass:")
print("-" * 50)

activations = [x]
for i, layer in enumerate(layers):
    out, _ = layer(activations[-1])
    activations.append(out)
    print(f"Layer {i}: input mean={activations[-2].abs().mean().item():.6e}, "
          f"output mean={out.abs().mean().item():.6e}")

# Loss at final output
loss = activations[-1][:, -1, :].sum()
print(f"\nLoss: {loss.item():.6f}")

# Backward
loss.backward()

print("\nBackward pass - gradient magnitudes:")
print("-" * 50)

# Check gradient at each layer's parameters
for i, layer in enumerate(layers):
    w_x_grad = layer.cell.W_x.grad.abs().mean().item() if layer.cell.W_x.grad is not None else 0
    w_out_grad = layer.cell.W_out.grad.abs().mean().item() if layer.cell.W_out.grad is not None else 0
    in_proj_grad = layer.in_proj.weight.grad.abs().mean().item() if layer.in_proj.weight.grad is not None else 0
    out_proj_grad = layer.out_proj.weight.grad.abs().mean().item() if layer.out_proj.weight.grad is not None else 0

    print(f"Layer {i}: W_x={w_x_grad:.6e}, W_out={w_out_grad:.6e}, "
          f"in_proj={in_proj_grad:.6e}, out_proj={out_proj_grad:.6e}")

print(f"\nInput gradient: {x.grad.abs().mean().item():.6e}")

# Now let's trace layer by layer to see where gradients die
print("\n" + "=" * 70)
print("LAYER-BY-LAYER GRADIENT ANALYSIS")
print("=" * 70)

# Fresh run with hooks
layers2 = nn.ModuleList([
    LogStorageDiagonal(dim=dim, expansion=1.0, n_groups=n_groups, delta_init=-2.0)
    for _ in range(n_layers)
]).to(device).to(dtype)

grad_outputs = {}

def make_hook(name):
    def hook(grad):
        grad_outputs[name] = grad.abs().mean().item()
        return grad
    return hook

x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

activations2 = [x2]
for i, layer in enumerate(layers2):
    out, _ = layer(activations2[-1])
    out.register_hook(make_hook(f"layer_{i}_output"))
    activations2.append(out)

loss2 = activations2[-1][:, -1, :].sum()
loss2.backward()

print("\nGradient at each layer's output:")
for i in range(n_layers):
    key = f"layer_{i}_output"
    if key in grad_outputs:
        print(f"  Layer {i} output grad: {grad_outputs[key]:.6e}")

print(f"  Input grad: {x2.grad.abs().mean().item():.6e}")

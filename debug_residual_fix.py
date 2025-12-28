"""
Test: Does adding residual connections fix the gradient death?
"""

import sys
sys.path.insert(0, '/home/erikg/haste_src')

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
dtype = torch.float32

dim = 256
seq_len = 64
batch = 2
n_groups = 16
n_layers = 4

print("=" * 70)
print("RESIDUAL CONNECTION FIX")
print("=" * 70)

class GRULayerWithResidual(nn.Module):
    def __init__(self, dim, n_groups=16):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.W_delta = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)
        self.r_h = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), -2.0))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, use_residual=True):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            x_t = x_proj[:, t, :]
            delta = torch.sigmoid(self.W_delta(x_t) + self.b_delta)
            candidate = torch.tanh(self.W_x(x_t) + self.r_h * h + self.b)
            h = (1 - delta) * h + delta * candidate

            # Output with compete
            w_out_h = self.W_out(h)
            h_grouped = h.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_grouped, dim=-1).view(B, D)
            out = compete * F.silu(w_out_h)
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = self.out_proj(output)

        if use_residual:
            output = output + x  # RESIDUAL!

        return output

# Test with residual
print("\n--- WITH COMPETE + RESIDUAL ---")
layers = nn.ModuleList([
    GRULayerWithResidual(dim, n_groups) for _ in range(n_layers)
]).to(device).to(dtype)

x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

activations = [x]
for layer in layers:
    out = layer(activations[-1], use_residual=True)
    activations.append(out)

loss = activations[-1][:, -1, :].sum()
loss.backward()

print(f"Output values per layer:")
for i, act in enumerate(activations):
    print(f"  Layer {i}: mean={act.abs().mean().item():.6e}")

print(f"\nGradient at each layer:")
for i, layer in enumerate(layers):
    grad = layer.W_x.weight.grad.abs().mean().item()
    print(f"  Layer {i}: W_x grad={grad:.6e}")

print(f"\nInput gradient: {x.grad.abs().mean().item():.6e}")

# Without residual for comparison
print("\n--- WITH COMPETE, NO RESIDUAL (original) ---")
layers2 = nn.ModuleList([
    GRULayerWithResidual(dim, n_groups) for _ in range(n_layers)
]).to(device).to(dtype)

x2 = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

activations2 = [x2]
for layer in layers2:
    out = layer(activations2[-1], use_residual=False)
    activations2.append(out)

loss2 = activations2[-1][:, -1, :].sum()
loss2.backward()

print(f"Output values per layer:")
for i, act in enumerate(activations2):
    print(f"  Layer {i}: mean={act.abs().mean().item():.6e}")

print(f"\nGradient at each layer:")
for i, layer in enumerate(layers2):
    grad = layer.W_x.weight.grad.abs().mean().item()
    print(f"  Layer {i}: W_x grad={grad:.6e}")

print(f"\nInput gradient: {x2.grad.abs().mean().item():.6e}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("Residual connections bypass the compete mechanism's gradient decay!")

"""Debug EXACT source of math difference."""
import torch
from mingru.perfect_gru import PerfectGRU
from mingru.standard_gru import StandardGRU

torch.manual_seed(42)

B, T, H = 2, 4, 8  # Small for easy inspection
print(f"Testing tiny model: B={B}, T={T}, H={H}\n")

# Create models
perfect_gru = PerfectGRU(dim=H, expansion_factor=1.0).cuda().float()
std_gru = StandardGRU(dim=H, expansion_factor=1.0).cuda().float()

# Copy weights
with torch.no_grad():
    perfect_gru.input_proj.weight.data.copy_(std_gru.input_proj.weight.data)
    perfect_gru.input_projection.weight.data.copy_(std_gru.gru.weight_ih_l0.data)
    perfect_gru.U_recurrent.weight.data.copy_(std_gru.gru.weight_hh_l0.data)
    perfect_gru.bias_ih.data.copy_(std_gru.gru.bias_ih_l0.data)
    perfect_gru.bias_hh.data.copy_(std_gru.gru.bias_hh_l0.data)
    perfect_gru.output_proj.weight.data.copy_(std_gru.output_proj.weight.data)

x = torch.randn(B, T, H, device='cuda', dtype=torch.float32)

# Run both forward passes
out_perfect, h_perfect, _ = perfect_gru(x)
out_std, h_std, _ = std_gru(x)

print("="*80)
print("FORWARD COMPARISON:")
print("="*80)

# Check inputs match
x_proj_perfect = perfect_gru.input_proj(x)
x_proj_std = std_gru.input_proj(x)
print(f"\n1. Input projection:")
print(f"   Diff: {(x_proj_perfect - x_proj_std).abs().max().item():.2e}")

# Check gate projections
gates_perfect = perfect_gru.input_projection(x_proj_perfect)
gates_std_manual = x_proj_std @ std_gru.gru.weight_ih_l0.t() + std_gru.gru.bias_ih_l0

print(f"\n2. Input gates (W_ih @ x + b_ih):")
print(f"   Diff: {(gates_perfect - gates_std_manual).abs().max().item():.2e}")

# Now manually step through ONE timestep
print(f"\n3. First timestep detailed:")
print("-"*80)

t = 0
h_prev = torch.zeros(B, H, device='cuda', dtype=torch.float32)

# Perfect GRU computation
x_t = gates_perfect[:, t, :]
h_rec_perfect = perfect_gru.U_recurrent(h_prev)

i_r, i_z, i_n = x_t.chunk(3, dim=1)
r_rec, z_rec, n_rec = h_rec_perfect.chunk(3, dim=1)
b_ih_r, b_ih_z, b_ih_n = perfect_gru.bias_ih.chunk(3)
b_hh_r, b_hh_z, b_hh_n = perfect_gru.bias_hh.chunk(3)

r_perfect = torch.sigmoid(i_r + r_rec + b_ih_r + b_hh_r)
z_perfect = torch.sigmoid(i_z + z_rec + b_ih_z + b_hh_z)
n_perfect = torch.tanh(i_n + b_ih_n + r_perfect * (n_rec + b_hh_n))
h_new_perfect = (1 - z_perfect) * n_perfect + z_perfect * h_prev

# Standard GRU computation (extract from nn.GRU)
x_proj_t = x_proj_std[:, t:t+1, :]  # [B, 1, H]
h0 = h_prev.unsqueeze(0)  # [1, B, H]
out_std_t, h_std_t = std_gru.gru(x_proj_t, h0)
h_new_std = h_std_t.squeeze(0)

print(f"Hidden state after t=0:")
print(f"  Perfect: {h_new_perfect[0, :4]}")
print(f"  Standard: {h_new_std[0, :4]}")
print(f"  Diff: {(h_new_perfect - h_new_std).abs().max().item():.2e}")

# Let me manually compute what nn.GRU does
print(f"\n4. Manual nn.GRU computation:")
print("-"*80)

# nn.GRU does: gates_i = x @ W_ih.T + b_ih, gates_h = h @ W_hh.T + b_hh
# Then splits and computes r, z, n

gi = x_proj_std[:, t, :] @ std_gru.gru.weight_ih_l0.t() + std_gru.gru.bias_ih_l0
gh = h_prev @ std_gru.gru.weight_hh_l0.t() + std_gru.gru.bias_hh_l0

gi_r, gi_z, gi_n = gi.chunk(3, dim=1)
gh_r, gh_z, gh_n = gh.chunk(3, dim=1)

r_manual = torch.sigmoid(gi_r + gh_r)
z_manual = torch.sigmoid(gi_z + gh_z)
n_manual = torch.tanh(gi_n + r_manual * gh_n)
h_manual = (1 - z_manual) * n_manual + z_manual * h_prev

print(f"  Manual r: {r_manual[0, :4]}")
print(f"  Perfect r: {r_perfect[0, :4]}")
print(f"  Diff: {(r_manual - r_perfect).abs().max().item():.2e}")

print(f"\n  Manual z: {z_manual[0, :4]}")
print(f"  Perfect z: {z_perfect[0, :4]}")
print(f"  Diff: {(z_manual - z_perfect).abs().max().item():.2e}")

print(f"\n  Manual n: {n_manual[0, :4]}")
print(f"  Perfect n: {n_perfect[0, :4]}")
print(f"  Diff: {(n_manual - n_perfect).abs().max().item():.2e}")

print(f"\n  Manual h: {h_manual[0, :4]}")
print(f"  Perfect h: {h_new_perfect[0, :4]}")
print(f"  Standard h: {h_new_std[0, :4]}")
print(f"  Diff (manual vs perfect): {(h_manual - h_new_perfect).abs().max().item():.2e}")
print(f"  Diff (manual vs standard): {(h_manual - h_new_std).abs().max().item():.2e}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if (h_manual - h_new_perfect).abs().max().item() < 1e-7:
    print("✅ Perfect GRU matches manual computation")
else:
    print("❌ Perfect GRU doesn't match manual - check implementation")

if (h_manual - h_new_std).abs().max().item() < 1e-7:
    print("✅ Standard GRU matches manual computation")
else:
    print("❌ Standard GRU doesn't match manual - unexpected!")

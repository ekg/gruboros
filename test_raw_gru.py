"""Compare PerfectGRU to raw PyTorch nn.GRU (no wrapping)."""
import torch
from mingru.perfect_gru import PerfectGRU

torch.manual_seed(42)

B, T, H = 2, 64, 8
print(f"Testing: B={B}, T={T}, H={H}")
print("="*80)

# Create raw nn.GRU (no wrapper)
raw_gru = torch.nn.GRU(
    input_size=H,
    hidden_size=H,
    num_layers=1,
    batch_first=True,
    bias=True
).cuda().float()

# Create PerfectGRU
perfect_gru = PerfectGRU(dim=H, expansion_factor=1.0).cuda().float()

# Copy weights from raw GRU to Perfect GRU
print("\nCopying weights...")
with torch.no_grad():
    # Perfect has: input_proj + input_projection + U_recurrent + biases
    # Raw GRU has: weight_ih, weight_hh, bias_ih, bias_hh

    # Set input_proj to identity (since we'll feed projected input)
    perfect_gru.input_proj.weight.data = torch.eye(H, device='cuda')

    # Copy GRU weights directly
    perfect_gru.input_projection.weight.data.copy_(raw_gru.weight_ih_l0.data)
    perfect_gru.U_recurrent.weight.data.copy_(raw_gru.weight_hh_l0.data)
    perfect_gru.bias_ih.data.copy_(raw_gru.bias_ih_l0.data)
    perfect_gru.bias_hh.data.copy_(raw_gru.bias_hh_l0.data)

    # Set output_proj to identity
    perfect_gru.output_proj.weight.data = torch.eye(H, device='cuda')

print("✓ Weights copied\n")

# Test with same input
x = torch.randn(B, T, H, device='cuda', dtype=torch.float32)

# Raw GRU forward
h0_raw = torch.zeros(1, B, H, device='cuda', dtype=torch.float32)
out_raw, hn_raw = raw_gru(x, h0_raw)

# Perfect GRU forward
out_perfect, hn_perfect, _ = perfect_gru(x)

print("COMPARISON:")
print("-"*80)

out_diff = (out_perfect - out_raw).abs()
h_diff = (hn_perfect - hn_raw.squeeze(0)).abs()

print(f"Output diff:")
print(f"  Max: {out_diff.max().item():.2e}")
print(f"  Mean: {out_diff.mean().item():.2e}")

print(f"\nHidden state diff:")
print(f"  Max: {h_diff.max().item():.2e}")
print(f"  Mean: {h_diff.mean().item():.2e}")

if out_diff.max().item() < 1e-6:
    print("\n✅ PERFECT MATCH - PerfectGRU == raw nn.GRU!")
else:
    print("\n❌ MISMATCH - investigating...")

    # Check first timestep in detail
    print("\nFirst timestep analysis:")
    print("-"*80)

    # Manually compute first timestep
    x_t = x[:, 0, :]  # [B, H]
    h_prev = torch.zeros(B, H, device='cuda', dtype=torch.float32)

    gi = x_t @ raw_gru.weight_ih_l0.t() + raw_gru.bias_ih_l0
    gh = h_prev @ raw_gru.weight_hh_l0.t() + raw_gru.bias_hh_l0

    gi_r, gi_z, gi_n = gi.chunk(3, dim=1)
    gh_r, gh_z, gh_n = gh.chunk(3, dim=1)

    r = torch.sigmoid(gi_r + gh_r)
    z = torch.sigmoid(gi_z + gh_z)
    n = torch.tanh(gi_n + r * gh_n)
    h_manual = (1 - z) * n + z * h_prev

    print(f"  Manual h[0, :4]: {h_manual[0, :4]}")
    print(f"  Raw GRU h[0, :4]: {out_raw[0, 0, :4]}")
    print(f"  Perfect h[0, :4]: {out_perfect[0, 0, :4]}")

    manual_vs_raw = (h_manual - out_raw[:, 0, :]).abs().max().item()
    manual_vs_perfect = (h_manual - out_perfect[:, 0, :]).abs().max().item()

    print(f"\n  Manual vs Raw: {manual_vs_raw:.2e}")
    print(f"  Manual vs Perfect: {manual_vs_perfect:.2e}")

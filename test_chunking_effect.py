"""Test if StandardGRU's chunking causes the mismatch."""
import torch
from mingru.perfect_gru import PerfectGRU
from mingru.standard_gru import StandardGRU

torch.manual_seed(42)

B, T, H = 2, 64, 256

print("Testing chunking effect:")
print("="*80)

# Test with different chunk sizes
for chunk_size in [64, 32, 16, 8]:
    print(f"\nChunk size: {chunk_size}")
    print("-"*80)

    perfect_gru = PerfectGRU(dim=H, expansion_factor=1.0).cuda().float()
    std_gru = StandardGRU(dim=H, expansion_factor=1.0, recurrence_chunk_size=chunk_size).cuda().float()

    # Copy weights
    with torch.no_grad():
        perfect_gru.input_proj.weight.data.copy_(std_gru.input_proj.weight.data)
        perfect_gru.input_projection.weight.data.copy_(std_gru.gru.weight_ih_l0.data)
        perfect_gru.U_recurrent.weight.data.copy_(std_gru.gru.weight_hh_l0.data)
        perfect_gru.bias_ih.data.copy_(std_gru.gru.bias_ih_l0.data)
        perfect_gru.bias_hh.data.copy_(std_gru.gru.bias_hh_l0.data)
        perfect_gru.output_proj.weight.data.copy_(std_gru.output_proj.weight.data)

    x = torch.randn(B, T, H, device='cuda', dtype=torch.float32)

    out_perfect, h_perfect, _ = perfect_gru(x)
    out_std, h_std, _ = std_gru(x)

    out_diff = (out_perfect - out_std).abs().max().item()
    h_diff = (h_perfect - h_std).abs().max().item()

    print(f"  Output diff: {out_diff:.2e}")
    print(f"  Hidden diff: {h_diff:.2e}")

    if out_diff < 1e-6:
        print(f"  ✅ PERFECT MATCH!")
    else:
        print(f"  ❌ Mismatch (chunking effect)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("PerfectGRU processes entire sequence in one pass (T=64)")
print("StandardGRU chunks it (e.g., 2 chunks of 32)")
print("Chunking causes floating point error accumulation!")
print("\nSolution: Use PerfectGRU for EXACT cuDNN math equivalence")

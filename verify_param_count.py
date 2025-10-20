#!/usr/bin/env python
"""Verify parameter count for HybridGRU architecture."""

def calculate_params(dim, depth, vocab_size, expansion=1.0, ff_mult=0.0, conv_kernel=4):
    """
    Calculate total parameters for minLM with HybridGRU.

    Args:
        dim: Hidden dimension
        depth: Number of layers
        vocab_size: Vocabulary size
        expansion: GRU expansion factor (dim_inner = dim * expansion)
        ff_mult: FFN multiplier (0.0 = no FFN)
        conv_kernel: Convolution kernel size
    """

    dim_inner = int(dim * expansion)

    # Embedding layers
    token_emb = vocab_size * dim
    pos_emb = 0  # minLM doesn't use positional embeddings
    output_proj = vocab_size * dim

    total_embedding = token_emb + pos_emb + output_proj

    print("="*80)
    print("PARAMETER COUNT CALCULATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  dim: {dim}")
    print(f"  depth: {depth}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  expansion: {expansion}")
    print(f"  dim_inner: {dim_inner}")
    print(f"  ff_mult: {ff_mult}")
    print(f"  conv_kernel: {conv_kernel}")

    print(f"\nEmbedding Layers:")
    print(f"  token_emb:   {vocab_size:>6} × {dim:>5} = {token_emb:>12,} params")
    print(f"  output_proj: {vocab_size:>6} × {dim:>5} = {output_proj:>12,} params")
    print(f"  Total embedding:                {total_embedding:>12,} params")

    # Per-layer parameters
    print(f"\nPer-Layer Parameters (HybridGRU):")

    # Convolution (optional)
    conv_params = 0
    if conv_kernel > 0:
        conv_params = dim * conv_kernel
        print(f"  conv:        {dim:>6} × {conv_kernel:>5} = {conv_params:>12,} params")

    # GRU components
    # input_proj: projects input to 3 × dim_inner (for r, z, n gates)
    input_proj = dim * 3 * dim_inner
    print(f"  input_proj:  {dim:>6} × {3 * dim_inner:>5} = {input_proj:>12,} params")

    # hidden_proj: projects hidden state to 3 × dim_inner
    hidden_proj = dim_inner * 3 * dim_inner
    print(f"  hidden_proj: {dim_inner:>6} × {3 * dim_inner:>5} = {hidden_proj:>12,} params")

    # to_out: projects dim_inner back to dim
    to_out = dim_inner * dim
    print(f"  to_out:      {dim_inner:>6} × {dim:>5} = {to_out:>12,} params")

    # LayerNorm (negligible, but count for completeness)
    norm_params = dim * 2  # gamma and beta
    print(f"  norm:        {dim:>6} × {2:>5} = {norm_params:>12,} params")

    # FFN (optional)
    ffn_params = 0
    if ff_mult > 0:
        dim_ff = int(dim * ff_mult)
        ffn_params = dim * dim_ff + dim_ff * dim  # up and down projections
        print(f"  ffn:         {dim:>6} × {dim_ff:>5} × 2 = {ffn_params:>12,} params")

    per_layer_total = conv_params + input_proj + hidden_proj + to_out + norm_params + ffn_params
    print(f"  Per-layer total:                {per_layer_total:>12,} params")

    # Total layer parameters
    total_layers = per_layer_total * depth
    print(f"\nTotal Layer Parameters ({depth} layers):")
    print(f"  {per_layer_total:,} × {depth} = {total_layers:>12,} params")

    # Grand total
    total_params = total_embedding + total_layers

    print(f"\n{'='*80}")
    print(f"TOTAL PARAMETERS: {total_params:>12,}")
    print(f"{'='*80}")

    # Breakdown percentages
    emb_pct = 100.0 * total_embedding / total_params
    layer_pct = 100.0 * total_layers / total_params

    print(f"\nBreakdown:")
    print(f"  Embeddings: {total_embedding:>12,} params ({emb_pct:>5.1f}%)")
    print(f"  Layers:     {total_layers:>12,} params ({layer_pct:>5.1f}%)")

    # Parameter distribution
    print(f"\nParameter distribution:")
    print(f"  Embeddings: {emb_pct:.1f}%")
    print(f"  Layers: {layer_pct:.1f}%")

    return total_params


def find_optimal_depth(target_params, dim, vocab_size, expansion=1.0, ff_mult=0.0, conv_kernel=4):
    """Find optimal depth to hit target parameter count."""

    print(f"\n{'='*80}")
    print(f"FINDING OPTIMAL DEPTH FOR {target_params/1e6:.0f}M PARAMETERS")
    print(f"{'='*80}")

    best_depth = 0
    best_diff = float('inf')

    for depth in range(1, 30):
        params = calculate_embedding_params(dim, vocab_size)
        params += calculate_layer_params(dim, expansion, ff_mult, conv_kernel) * depth

        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_depth = depth

            if diff < target_params * 0.05:  # Within 5%
                break

    print(f"\nOptimal depth: {best_depth}")
    print(f"Estimated parameters: {params:,}")
    print(f"Target: {target_params:,}")
    print(f"Difference: {params - target_params:,} ({100.0 * (params - target_params) / target_params:+.1f}%)")

    return best_depth


def calculate_embedding_params(dim, vocab_size):
    """Calculate embedding parameters."""
    return vocab_size * dim * 2  # token_emb + output_proj


def calculate_layer_params(dim, expansion, ff_mult, conv_kernel):
    """Calculate per-layer parameters."""
    dim_inner = int(dim * expansion)

    conv_params = dim * conv_kernel if conv_kernel > 0 else 0
    input_proj = dim * 3 * dim_inner
    hidden_proj = dim_inner * 3 * dim_inner
    to_out = dim_inner * dim
    norm_params = dim * 2

    ffn_params = 0
    if ff_mult > 0:
        dim_ff = int(dim * ff_mult)
        ffn_params = dim * dim_ff * 2

    return conv_params + input_proj + hidden_proj + to_out + norm_params + ffn_params


if __name__ == "__main__":
    # HybridGRU configuration
    print("\n" + "="*80)
    print("HYBRIDGRU CONFIGURATION (from train.gru.sh)")
    print("="*80)

    dim = 1536
    depth = 14
    vocab_size = 100277  # TikToken cl100k_base
    expansion = 1.0
    ff_mult = 0.0  # NO FFN
    conv_kernel = 4

    total = calculate_params(dim, depth, vocab_size, expansion, ff_mult, conv_kernel)

    # Find optimal depth for exactly 500M
    print("\n" + "="*80)
    optimal_depth = find_optimal_depth(
        target_params=500_000_000,
        dim=dim,
        vocab_size=vocab_size,
        expansion=expansion,
        ff_mult=ff_mult,
        conv_kernel=conv_kernel
    )

    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    if abs(total - 500_000_000) > 50_000_000:
        print(f"Current depth={depth} gives {total/1e6:.1f}M params")
        print(f"For closer to 500M, use depth={optimal_depth}")

        # Calculate exact params with optimal depth
        print(f"\nRecalculating with depth={optimal_depth}:")
        calculate_params(dim, optimal_depth, vocab_size, expansion, ff_mult, conv_kernel)
    else:
        print(f"Current configuration is good: {total/1e6:.1f}M params ≈ 500M")

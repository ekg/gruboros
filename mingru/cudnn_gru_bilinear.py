"""
cuDNN GRU with True Bilinear (Second-Order) Interactions.

Key insight from multiplicative RNN research (Sutskever 2011, mLSTM 2016):
- Standard GRUs: h_t = f(W_h·h + W_x·x) - first-order only
- Multiplicative RNNs: h_t = f(W[x]·h) where W depends on input
- The second-order terms (h_i × x_j) enable XOR-like reasoning

This implementation adds true bilinear interactions to cuDNN GRU:
  h_gru = cuDNN_GRU(x)                    # Fast base sequence modeling
  factor_x = W_x @ x                      # Input projection
  factor_h = W_h @ h_gru                  # Hidden projection
  bilinear = factor_x * factor_h          # Second-order interaction!
  output = h_gru + W_out @ bilinear       # Residual connection

The element-wise product creates terms like (w_xi · x_i) × (w_hj · h_j),
which are exactly the bilinear interactions that enable complex reasoning.

References:
- Sutskever et al. (2011) "Generating Text with Recurrent Neural Networks"
- Krause et al. (2016) "Multiplicative LSTM for Sequence Modelling"
- Wu et al. (2016) "On Multiplicative Integration with RNNs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CuDNNGRU_Bilinear(nn.Module):
    """
    cuDNN GRU with bilinear (second-order) interactions.

    This creates true h×x terms that enable XOR-like reasoning:
    - Run cuDNN GRU for fast base sequence modeling
    - Add bilinear term: (W_x·x) ⊙ (W_h·h)
    - Merge back with residual connection

    The bilinear term creates second-order interactions between
    input content and hidden state that standard GRUs cannot express.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        bilinear_rank: int = None,  # Rank of bilinear factorization (None = full dim)
        use_gated_bilinear: bool = True,  # Gate the bilinear term
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs  # Ignore other params for API compat
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.bilinear_rank = bilinear_rank or self.dim_inner
        self.use_gated_bilinear = use_gated_bilinear

        # === cuDNN GRU for fast base sequence modeling ===
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # === Bilinear factorization ===
        # bilinear_term = (W_x @ x) * (W_h @ h)
        # This creates second-order terms: Σ_ij (w_xi·x_i)·(w_hj·h_j)
        self.W_x = nn.Linear(dim, self.bilinear_rank, bias=False)
        self.W_h = nn.Linear(self.dim_inner, self.bilinear_rank, bias=False)

        # Optional gating for the bilinear term
        if use_gated_bilinear:
            self.bilinear_gate = nn.Linear(self.bilinear_rank, self.bilinear_rank, bias=True)

        # Project bilinear back to model dimension
        self.W_out = nn.Linear(self.bilinear_rank, dim, bias=False)

        # Output projection if expansion != 1
        if expansion_factor != 1.0:
            self.gru_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.gru_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Initialize bilinear projections with small values
        nn.init.normal_(self.W_x.weight, std=0.02)
        nn.init.normal_(self.W_h.weight, std=0.02)

        # Small init for output to start as residual
        nn.init.normal_(self.W_out.weight, std=0.02 / math.sqrt(2))

        if self.use_gated_bilinear:
            nn.init.normal_(self.bilinear_gate.weight, std=0.02)
            # Initialize gate bias to 0 (sigmoid(0) = 0.5, neutral)
            nn.init.zeros_(self.bilinear_gate.bias)

        if not isinstance(self.gru_out, nn.Identity):
            nn.init.normal_(self.gru_out.weight, std=0.02)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass with bilinear interactions.

        Args:
            x: [B, T, D] input
            prev_hidden: [B, dim_inner] previous GRU hidden state
            return_next_prev_hidden: if True, return (output, next_hidden)

        Returns:
            output: [B, T, D]
            or (output, next_hidden) if return_next_prev_hidden
        """
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # === cuDNN GRU ===
        gru_dtype = self.gru.weight_ih_l0.dtype
        x_gru = x.to(gru_dtype).contiguous()

        if prev_hidden is not None:
            if prev_hidden.dim() == 2:
                h0 = prev_hidden.to(gru_dtype).unsqueeze(0).contiguous()
            else:
                h0 = prev_hidden.to(gru_dtype).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=gru_dtype)

        # Run GRU
        h_seq, h_final = self.gru(x_gru, h0)
        # h_seq: [B, T, dim_inner]
        # h_final: [1, B, dim_inner]

        h_seq = h_seq.to(dtype)
        h_final = h_final.squeeze(0).to(dtype)

        # === Bilinear Interaction ===
        # This is the key innovation: create second-order h×x terms
        factor_x = self.W_x(x)       # [B, T, bilinear_rank] - input contribution
        factor_h = self.W_h(h_seq)   # [B, T, bilinear_rank] - hidden contribution

        # Element-wise product creates bilinear terms
        # Each output dimension j has: Σ_i (w_xi·x_i) × (w_hi·h_i)
        bilinear = factor_x * factor_h  # [B, T, bilinear_rank]

        # Optional gating
        if self.use_gated_bilinear:
            gate = torch.sigmoid(self.bilinear_gate(bilinear))
            bilinear = bilinear * gate

        # Project bilinear term back to model dimension
        bilinear_out = self.W_out(bilinear)  # [B, T, dim]

        # === Combine with residual ===
        gru_out = self.gru_out(h_seq)  # [B, T, dim]
        out = gru_out + bilinear_out   # Residual connection

        if return_next_prev_hidden:
            return out, h_final
        return out


class CuDNNGRU_BilinearLM(nn.Module):
    """
    Language model using cuDNN GRU with bilinear interactions.

    This tests whether true second-order (h×x) interactions improve
    language modeling compared to standard GRU or post-hoc gating.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
        bilinear_rank: int = None,
        use_gated_bilinear: bool = True,
        ff_mult: float = 0.0,  # Optional FFN (0 = disabled)
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # GRU + Bilinear layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList() if ff_mult > 0 else None
        self.ff_norms = nn.ModuleList() if ff_mult > 0 else None

        for i in range(depth):
            self.layers.append(
                CuDNNGRU_Bilinear(
                    dim=dim,
                    expansion_factor=expansion_factor,
                    bilinear_rank=bilinear_rank,
                    use_gated_bilinear=use_gated_bilinear,
                    layer_idx=i,
                    num_layers=depth,
                )
            )
            self.norms.append(nn.LayerNorm(dim))

            if ff_mult > 0:
                self.ff_layers.append(nn.Sequential(
                    nn.Linear(dim, int(dim * ff_mult), bias=False),
                    nn.GELU(),
                    nn.Linear(int(dim * ff_mult), dim, bias=False),
                ))
                self.ff_norms.append(nn.LayerNorm(dim))

        # Output
        self.norm_f = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if self.ff_layers is not None:
            for ff in self.ff_layers:
                nn.init.normal_(ff[0].weight, std=0.02)
                nn.init.normal_(ff[2].weight, std=0.02 / math.sqrt(2 * self.depth))

    def forward(
        self,
        x,
        prev_hidden=None,
        prev_hiddens=None,
        prev_conv_buffers=None,
        return_next_prev_hidden=False,
        return_prev_hiddens=False,
        return_loss=False,
        doc_boundaries=None,
        **kwargs,
    ):
        """
        Forward pass matching train.py interface.
        """
        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Process through layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # GRU + Bilinear with residual
            h = h + layer(norm(h))

            # Optional FFN
            if self.ff_layers is not None:
                h = h + self.ff_layers[i](self.ff_norms[i](h))

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1),
        )

        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # No TBPTT state for now
        return loss, (None, None)


def count_parameters(model):
    """Count and breakdown parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if 'token_emb' in n)
    gru = sum(p.numel() for n, p in model.named_parameters() if 'gru' in n and 'gru_out' not in n)
    bilinear = sum(p.numel() for n, p in model.named_parameters() if 'W_x' in n or 'W_h' in n or 'W_out' in n or 'bilinear' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff' in n)

    return {
        'total': total,
        'embedding': embedding,
        'gru': gru,
        'bilinear': bilinear,
        'ff': ff,
    }


if __name__ == "__main__":
    print("Testing CuDNNGRU_Bilinear...")
    print("=" * 60)

    # Test single layer
    layer = CuDNNGRU_Bilinear(
        dim=256,
        expansion_factor=1.0,
        bilinear_rank=256,
        use_gated_bilinear=True,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Test full LM - find ~1B config
    print("\n" + "=" * 60)
    print("Finding ~1B params configuration:")

    for depth in [20, 24, 28, 32]:
        model = CuDNNGRU_BilinearLM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            expansion_factor=1.0,
            bilinear_rank=2048,  # Full rank
            use_gated_bilinear=True,
            ff_mult=0.0,  # No FFN for first test
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")
        print(f"  Bilinear: {counts['bilinear']:,} ({counts['bilinear']/1e6:.1f}M)")

        del model

    # Test forward/backward on small model
    print("\n" + "=" * 60)
    print("Testing forward/backward...")

    model = CuDNNGRU_BilinearLM(
        num_tokens=50280,
        dim=2048,
        depth=24,  # ~1B params
        expansion_factor=1.0,
        bilinear_rank=2048,
        use_gated_bilinear=True,
        ff_mult=0.0,
    ).cuda().bfloat16()

    counts = count_parameters(model)
    print(f"Model: {counts['total']:,} params ({counts['total']/1e9:.2f}B)")

    x = torch.randint(0, 50280, (2, 256), device='cuda')
    loss = model(x)
    print(f"Forward: loss={loss.item():.4f}")

    loss.backward()
    print("Backward succeeded!")

    print("\nDone!")

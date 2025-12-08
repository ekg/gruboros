"""
Plain cuDNN GRU baseline - no modifications whatsoever.

This is the simplest possible cuDNN GRU language model for establishing
a baseline. Just stack of:
  - LayerNorm
  - cuDNN GRU
  - Residual connection

No EMA, no bilinear terms, no multiplicative gates, no FFN.
Just pure GRU + residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PlainCuDNNGRU(nn.Module):
    """
    Plain cuDNN GRU layer with residual connection.

    No modifications - just GRU + output projection + residual.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs  # Ignore other params for API compat
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # cuDNN GRU
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Output projection if expansion != 1
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Small output init for residual start
        if not isinstance(self.to_out, nn.Identity):
            nn.init.normal_(self.to_out.weight, std=0.02)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass.

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

        # cuDNN GRU doesn't support bfloat16, use float32 or model dtype
        gru_dtype = self.gru.weight_ih_l0.dtype
        x_gru = x.to(gru_dtype).contiguous()

        # Prepare initial hidden state: [1, B, dim_inner]
        if prev_hidden is not None:
            if prev_hidden.dim() == 2:
                h0 = prev_hidden.to(gru_dtype).unsqueeze(0).contiguous()
            else:
                h0 = prev_hidden.to(gru_dtype).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=gru_dtype)

        # Run cuDNN GRU
        h_seq, h_final = self.gru(x_gru, h0)
        # h_seq: [B, T, dim_inner]
        # h_final: [1, B, dim_inner]

        h_seq = h_seq.to(dtype)
        h_final = h_final.squeeze(0).to(dtype)  # [B, dim_inner]

        # Output projection
        out = self.to_out(h_seq)

        if return_next_prev_hidden:
            return out, h_final
        return out


class PlainCuDNNGRU_LM(nn.Module):
    """
    Plain cuDNN GRU Language Model.

    Just embedding + stacked GRU layers with residual + output projection.
    No FFN, no EMA, no modifications - pure GRU baseline.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
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

        # GRU layers with pre-norm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList() if ff_mult > 0 else None
        self.ff_norms = nn.ModuleList() if ff_mult > 0 else None

        for i in range(depth):
            self.layers.append(
                PlainCuDNNGRU(
                    dim=dim,
                    expansion_factor=expansion_factor,
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
            # GRU with pre-norm and residual
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
    gru = sum(p.numel() for n, p in model.named_parameters() if 'gru' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff' in n)

    return {
        'total': total,
        'embedding': embedding,
        'gru': gru,
        'ff': ff,
    }


if __name__ == "__main__":
    print("Testing PlainCuDNNGRU...")
    print("=" * 60)

    # Test single layer
    layer = PlainCuDNNGRU(
        dim=256,
        expansion_factor=1.0,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Find ~1B config
    print("\n" + "=" * 60)
    print("Finding ~1B params configuration:")

    # GRU params per layer = 3 * (dim * dim + dim * dim + 2*dim) = 6*dim^2 + 6*dim ≈ 6*dim^2
    # For dim=2048: ~25M per layer
    # For 1B params with ~100M embedding: need ~900M in GRU
    # ~36 layers

    for depth in [32, 36, 40]:
        model = PlainCuDNNGRU_LM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            expansion_factor=1.0,
            ff_mult=0.0,  # No FFN
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")

        del model

    # Test forward/backward on target config
    print("\n" + "=" * 60)
    print("Testing forward/backward on ~1B model...")

    model = PlainCuDNNGRU_LM(
        num_tokens=50280,
        dim=2048,
        depth=36,  # ~1B params
        expansion_factor=1.0,
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

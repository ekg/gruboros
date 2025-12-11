"""
cuDNN GRU + 3-layer FFN Gate (selectivity mechanism).

Key insight from ablations:
- Both x and h matter for gating (Mult GRU 3.00 > input-only 3.06)
- Conv1d doesn't help
- Deeper selectivity network may learn better patterns

Architecture:
  h = cuDNN_GRU(x)                    # Global recurrence
  gate_input = concat([x, h])         # Both matter!
  gate = FFN3(gate_input)             # 3-layer deep selectivity
  h' = h * gate                       # Multiplicative interaction
  out = W_out @ h'                    # Output projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FFN3Gate(nn.Module):
    """3-layer FFN for computing the gate (selectivity mechanism)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # 3-layer FFN: input -> hidden -> hidden -> output
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        # Small init for stable training
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.normal_(self.fc3.weight, std=0.02 / math.sqrt(3))  # Scale down last layer

    def forward(self, x):
        # x: [B, T, input_dim]
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x) + self.gate_bias
        return torch.sigmoid(x)


class CuDNNGRU_FFN3(nn.Module):
    """
    cuDNN GRU with 3-layer FFN selectivity gate.

    Like Mamba2's selective mechanism but:
    - Uses cuDNN GRU for global recurrence (not SSM)
    - Uses 3-layer FFN on [x, h] for selectivity (not conv + linear)
    - Multiplicative gating on h
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ffn_expand: float = 2.0,  # Hidden dim multiplier for FFN3
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ffn_hidden = int(dim * ffn_expand)

        # === cuDNN GRU ===
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # === 3-layer FFN Gate ===
        # Input: concat([x, h]) = dim + dim_inner
        # Output: dim_inner (to match h for elementwise mult)
        self.gate_ffn = FFN3Gate(
            input_dim=dim + self.dim_inner,
            hidden_dim=self.ffn_hidden,
            output_dim=self.dim_inner
        )

        # === Output projection ===
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
            nn.init.normal_(self.to_out.weight, std=0.02 / math.sqrt(2))
        else:
            self.to_out = nn.Identity()

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass with GRU + FFN3 gate.

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

        h_seq, h_final = self.gru(x_gru, h0)
        h_seq = h_seq.to(dtype)
        h_final = h_final.squeeze(0).to(dtype)

        # === 3-layer FFN Gate ===
        # Concatenate x and h for gate computation
        gate_input = torch.cat([x, h_seq], dim=-1)  # [B, T, dim + dim_inner]
        gate = self.gate_ffn(gate_input)  # [B, T, dim_inner]

        # Multiplicative gating
        h_gated = h_seq * gate

        # === Output ===
        out = self.to_out(h_gated)

        if return_next_prev_hidden:
            return out, h_final
        return out


class CuDNNGRU_FFN3_LM(nn.Module):
    """
    Language model using cuDNN GRU + 3-layer FFN selectivity gate.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
        ffn_expand: float = 2.0,
        ff_mult: float = 0.0,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList() if ff_mult > 0 else None
        self.ff_norms = nn.ModuleList() if ff_mult > 0 else None

        for i in range(depth):
            self.layers.append(
                CuDNNGRU_FFN3(
                    dim=dim,
                    expansion_factor=expansion_factor,
                    ffn_expand=ffn_expand,
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
        """Forward pass matching train.py interface."""
        h = self.token_emb(x)
        h = self.drop(h)

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = h + layer(norm(h))

            if self.ff_layers is not None:
                h = h + self.ff_layers[i](self.ff_norms[i](h))

        h = self.norm_f(h)
        logits = self.to_logits(h)

        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1),
        )

        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        return loss, (None, None)


def count_parameters(model):
    """Count and breakdown parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if 'token_emb' in n)
    gru = sum(p.numel() for n, p in model.named_parameters() if 'gru' in n)
    gate_ffn = sum(p.numel() for n, p in model.named_parameters() if 'gate_ffn' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff_layers' in n)

    return {
        'total': total,
        'embedding': embedding,
        'gru': gru,
        'gate_ffn': gate_ffn,
        'ff': ff,
    }


if __name__ == "__main__":
    print("Testing CuDNNGRU_FFN3...")
    print("=" * 60)

    # Test single layer
    layer = CuDNNGRU_FFN3(
        dim=256,
        expansion_factor=1.0,
        ffn_expand=2.0,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Count FFN3 gate params
    gate_params = sum(p.numel() for p in layer.gate_ffn.parameters())
    print(f"FFN3 gate params: {gate_params:,}")

    # Test full LM - find ~1B config
    print("\n" + "=" * 60)
    print("Finding ~1B params configuration:")

    for depth in [20, 22, 24, 26]:
        model = CuDNNGRU_FFN3_LM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            expansion_factor=1.0,
            ffn_expand=2.0,
            ff_mult=0.0,
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")
        print(f"  FFN3 gate: {counts['gate_ffn']:,} ({counts['gate_ffn']/1e6:.1f}M)")

        del model

    # Test forward/backward
    print("\n" + "=" * 60)
    print("Testing forward/backward...")

    model = CuDNNGRU_FFN3_LM(
        num_tokens=50280,
        dim=2048,
        depth=22,  # ~1B params
        expansion_factor=1.0,
        ffn_expand=2.0,
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

"""
Elman Selective: Simple MLP Recurrence + SiLU Output Selectivity

The simplest possible recurrent architecture:
1. Concatenate input x and hidden state h
2. Push through MLP with tanh (state evolution)
3. Apply SiLU selectivity gate (output filtering)

Hypothesis: GRU/LSTM gates are overengineered. A simple MLP can handle
state mixing - what matters is selectivity on the output.

Architecture per layer:
    combined = concat([x, h], dim=-1)           # [2*dim]
    hidden = tanh(W1 @ combined)                # [expansion*dim]
    h_new = W2 @ hidden                         # [dim]
    gate = silu(Wg_x @ x + Wg_h @ h_new)        # Selectivity
    out = h_new * gate

This is essentially Elman (1990) + depth + output selectivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ElmanSelectiveLayer(nn.Module):
    """
    Single layer: MLP state mixing + SiLU selectivity.

    The MLP mixes [x, h] to produce new hidden state.
    The selectivity gate filters what gets output.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        layer_idx: int = None,
        num_layers: int = None,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(dim * expansion)

        # Elman MLP: concat([x, h]) → tanh → h_new
        self.W1 = nn.Linear(dim * 2, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, dim, bias=False)

        # SiLU selectivity gate
        self.gate_x = nn.Linear(dim, dim, bias=False)
        self.gate_h = nn.Linear(dim, dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(dim))

        self._init_weights(layer_idx, num_layers)

    def _init_weights(self, layer_idx, num_layers):
        # Standard initialization for MLP
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.normal_(self.W2.weight, std=0.02)

        # Gate initialization
        nn.init.normal_(self.gate_x.weight, std=0.02)
        nn.init.normal_(self.gate_h.weight, std=0.02)

    def forward(self, x, h_prev=None):
        """
        Args:
            x: [B, T, dim] input at this layer
            h_prev: [B, dim] hidden state from previous timestep (or None)

        Returns:
            out: [B, T, dim] layer output
            h_final: [B, dim] final hidden state
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden state if needed
        if h_prev is None:
            h = torch.zeros(B, D, device=device, dtype=dtype)
        else:
            h = h_prev

        # Process sequentially through time
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # [B, dim]

            # Elman MLP mixing
            combined = torch.cat([x_t, h], dim=-1)  # [B, 2*dim]
            hidden = torch.tanh(self.W1(combined))   # [B, hidden_dim]
            h_new = self.W2(hidden)                  # [B, dim]

            # SiLU selectivity gate
            gate = F.silu(self.gate_x(x_t) + self.gate_h(h_new) + self.gate_bias)
            out_t = h_new * gate

            outputs.append(out_t)
            h = h_new  # Update hidden state

        out = torch.stack(outputs, dim=1)  # [B, T, dim]
        return out, h


class ElmanSelectiveLM(nn.Module):
    """
    Language model using Elman MLP + Selectivity.

    Structure (same as our other models):
    - Token embedding
    - N layers of: LayerNorm → ElmanMLP → Selectivity → Residual
    - Final LayerNorm
    - Output projection (tied weights)
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Elman + Selectivity layers
        self.layers = nn.ModuleList([
            ElmanSelectiveLayer(
                dim=dim,
                expansion=expansion,
                layer_idx=i,
                num_layers=depth,
            )
            for i in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])

        # Output
        self.norm_f = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
        self,
        x,
        prev_hiddens=None,
        return_prev_hiddens=False,
        return_loss=True,
        **kwargs,
    ):
        """Forward pass matching train.py interface."""
        B, T = x.shape

        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Process through layers (no TBPTT - each chunk independent)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            layer_out, _ = layer(norm(h), h_prev=None)
            h = h + layer_out  # Residual connection

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1),
        )

        if return_prev_hiddens:
            return loss, (None, None)
        return loss


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("ElmanSelective Parameter Count Test")
    print("=" * 60)

    # Test different configurations to find ~1B params
    dim = 2048
    num_tokens = 50281

    print(f"\nWith dim={dim}, num_tokens={num_tokens}:")
    print("-" * 40)

    for depth in [18, 20, 22, 24, 27]:
        for expansion in [2.0, 2.5, 3.0]:
            model = ElmanSelectiveLM(
                num_tokens=num_tokens,
                dim=dim,
                depth=depth,
                expansion=expansion,
            )
            params = count_parameters(model)
            marker = " <-- ~1B" if 0.95e9 < params < 1.05e9 else ""
            print(f"depth={depth}, expansion={expansion}: {params:,} ({params/1e9:.3f}B){marker}")
            del model

    # Test forward/backward with target config
    print("\n" + "=" * 60)
    print("Forward/Backward Test")
    print("=" * 60)

    # Find best config for ~1B
    model = ElmanSelectiveLM(
        num_tokens=50281,
        dim=2048,
        depth=27,
        expansion=2.0,
    ).cuda().bfloat16()

    params = count_parameters(model)
    print(f"Model: {params:,} params ({params/1e9:.3f}B)")

    # Test forward
    x = torch.randint(0, 50281, (2, 128), device='cuda')
    loss = model(x)
    print(f"Forward: loss={loss.item():.4f}")

    # Test backward
    loss.backward()
    print("Backward: OK")

    print("\nDone!")

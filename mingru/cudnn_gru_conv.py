"""
cuDNN GRU + Causal Conv1d for local context (like Mamba2).

Key insight from Mamba2:
- Mamba2 uses a 4-wide causal conv1d BEFORE the SSM layer
- This gives local context to help the recurrence
- Our ablation showed input-dependent gating (3.06) beat h-dependent (3.16)
- Adding conv1d gives the GRU richer local input features

Architecture:
  x_conv = CausalConv1d(x, kernel_size=4)  # Local context
  h = cuDNN_GRU(x_conv)                     # Sequence mixing
  gate = sigma(W_x @ x + W_h @ h + b)       # Content+state-dependent gate
  h' = h * gate                             # Multiplicative interaction
  out = W_out @ h'                          # Output projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    """Causal 1D convolution (like Mamba2 uses)."""

    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Pad left for causal
            groups=dim,  # Depthwise like Mamba2
            bias=bias,
        )

    def forward(self, x, conv_state=None, return_state=False):
        """
        Args:
            x: [B, T, D] input
            conv_state: [B, D, kernel_size-1] previous conv state for inference
            return_state: if True, return (output, new_state)
        Returns:
            output: [B, T, D]
        """
        B, T, D = x.shape

        # x: [B, T, D] -> [B, D, T] for conv1d
        x_t = x.transpose(1, 2)

        if conv_state is not None:
            # Inference mode: prepend previous state
            x_t = torch.cat([conv_state, x_t], dim=-1)

        # Apply conv (with left padding from nn.Conv1d padding=kernel_size-1)
        y = self.conv(x_t)

        # Remove extra elements to get back to original T length
        # Conv output is [B, D, T + kernel_size - 1], take first T elements
        y = y[..., :T]

        # Back to [B, T, D]
        y = y.transpose(1, 2)

        if return_state:
            # New state is last (kernel_size-1) positions of input
            new_state = x_t[..., -(self.kernel_size - 1):]
            return y, new_state
        return y


class CuDNNGRU_Conv(nn.Module):
    """
    cuDNN GRU with causal conv1d for local context (like Mamba2).

    Adds a 4-wide causal conv1d before the GRU:
    1. CausalConv1d gives each position access to 4 previous tokens
    2. GRU does global sequence mixing on the locally-enriched features
    3. Multiplicative gate creates x-h interactions
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,
        conv_kernel: int = 4,  # Like Mamba2
        use_input_gate: bool = True,
        use_hidden_gate: bool = True,
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.gate_dim = int(self.dim_inner * gate_expansion)
        self.use_input_gate = use_input_gate
        self.use_hidden_gate = use_hidden_gate
        self.conv_kernel = conv_kernel

        # === Causal Conv1d (like Mamba2) ===
        self.conv = CausalConv1d(dim, kernel_size=conv_kernel)

        # === cuDNN GRU ===
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # === Multiplicative Gate ===
        if use_input_gate:
            self.gate_x = nn.Linear(dim, self.gate_dim, bias=False)
        else:
            self.gate_x = None

        if use_hidden_gate:
            self.gate_h = nn.Linear(self.dim_inner, self.gate_dim, bias=False)
        else:
            self.gate_h = None

        self.gate_bias = nn.Parameter(torch.zeros(self.gate_dim))

        if self.gate_dim != self.dim_inner:
            self.gate_proj = nn.Linear(self.gate_dim, self.dim_inner, bias=False)
        else:
            self.gate_proj = None

        # === Output projection ===
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Conv init
        nn.init.normal_(self.conv.conv.weight, std=0.02)
        if self.conv.conv.bias is not None:
            nn.init.zeros_(self.conv.conv.bias)

        # Gate init
        if self.gate_x is not None:
            nn.init.normal_(self.gate_x.weight, std=0.02)
        if self.gate_h is not None:
            nn.init.normal_(self.gate_h.weight, std=0.02)
        if self.gate_proj is not None:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

        # Output projection
        if not isinstance(self.to_out, nn.Identity):
            nn.init.normal_(self.to_out.weight, std=0.02 / math.sqrt(2))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass with conv1d + GRU + multiplicative gate.

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

        # === Causal Conv1d (local context like Mamba2) ===
        x_conv = self.conv(x)

        # === cuDNN GRU ===
        gru_dtype = self.gru.weight_ih_l0.dtype
        x_gru = x_conv.to(gru_dtype).contiguous()

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

        # === Multiplicative Gate ===
        gate_logits = self.gate_bias.view(1, 1, -1)

        if self.gate_x is not None:
            gate_logits = gate_logits + self.gate_x(x)  # Use original x, not conv

        if self.gate_h is not None:
            gate_logits = gate_logits + self.gate_h(h_seq)

        gate = torch.sigmoid(gate_logits)

        if self.gate_proj is not None:
            gate = self.gate_proj(gate)

        h_gated = h_seq * gate

        # === Output ===
        out = self.to_out(h_gated)

        if return_next_prev_hidden:
            return out, h_final
        return out


class CuDNNGRU_ConvLM(nn.Module):
    """
    Language model using cuDNN GRU + Conv1d + Multiplicative Gating.

    Like Mamba2: causal conv1d before the recurrent layer for local context.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,
        conv_kernel: int = 4,
        use_input_gate: bool = True,
        use_hidden_gate: bool = True,
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
                CuDNNGRU_Conv(
                    dim=dim,
                    expansion_factor=expansion_factor,
                    gate_expansion=gate_expansion,
                    conv_kernel=conv_kernel,
                    use_input_gate=use_input_gate,
                    use_hidden_gate=use_hidden_gate,
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
    conv = sum(p.numel() for n, p in model.named_parameters() if 'conv' in n)
    gate = sum(p.numel() for n, p in model.named_parameters() if 'gate' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff' in n)

    return {
        'total': total,
        'embedding': embedding,
        'gru': gru,
        'conv': conv,
        'gate': gate,
        'ff': ff,
    }


if __name__ == "__main__":
    print("Testing CuDNNGRU_Conv...")
    print("=" * 60)

    # Test single layer
    layer = CuDNNGRU_Conv(
        dim=256,
        expansion_factor=1.0,
        gate_expansion=1.0,
        conv_kernel=4,
        use_input_gate=True,
        use_hidden_gate=True,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Count conv params
    conv_params = sum(p.numel() for p in layer.conv.parameters())
    print(f"Conv params: {conv_params:,}")

    # Test full LM - find ~1B config
    print("\n" + "=" * 60)
    print("Finding ~1B params configuration:")

    for depth in [24, 26, 27, 28]:
        model = CuDNNGRU_ConvLM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            expansion_factor=1.0,
            gate_expansion=1.0,
            conv_kernel=4,
            use_input_gate=True,
            use_hidden_gate=True,
            ff_mult=0.0,
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")
        print(f"  Conv: {counts['conv']:,} ({counts['conv']/1e6:.1f}M)")
        print(f"  Gate: {counts['gate']:,} ({counts['gate']/1e6:.1f}M)")

        del model

    # Test forward/backward
    print("\n" + "=" * 60)
    print("Testing forward/backward...")

    model = CuDNNGRU_ConvLM(
        num_tokens=50280,
        dim=2048,
        depth=27,  # ~1B params
        expansion_factor=1.0,
        gate_expansion=1.0,
        conv_kernel=4,
        use_input_gate=True,
        use_hidden_gate=True,
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

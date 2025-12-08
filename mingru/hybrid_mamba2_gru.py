"""
Hybrid Mamba2 + cuDNN GRU: Parallel paths with learned mixing.

Architecture per layer:
    h_mamba = Mamba2(norm(x))       # Linear SSM path
    h_gru = cuDNN_GRU(norm(x))      # Nonlinear GRU path
    output = x + W_m @ h_mamba + W_g @ h_gru  # Residual + learned mix

Key design choices:
- No TBPTT: Process full chunks independently (like Mamba2)
- No EMA: Just pure cuDNN GRU for the nonlinear path
- Parallel execution: Model learns when to use which path
- FF expansion on both paths for capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Hybrid model unavailable.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class HybridMamba2GRUBlock(nn.Module):
    """
    Single hybrid block with Mamba2 and cuDNN GRU in parallel.

    Both paths process the same input, outputs are linearly combined.
    This lets the model learn when to use the linear SSM vs nonlinear GRU.
    """

    def __init__(
        self,
        dim: int,
        # Mamba2 params
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        # GRU params
        gru_expansion: float = 1.0,  # Hidden size = dim * expansion
        # General
        ff_mult: float = 0.0,  # No FFN to match Mamba2 structure
        layer_idx: int = None,
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # === Shared pre-norm ===
        self.norm = RMSNorm(dim)

        # === Mamba2 path ===
        self.mamba2 = Mamba2(
            d_model=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            layer_idx=layer_idx,
        )

        # === cuDNN GRU path ===
        # Note: cuDNN GRU doesn't support bfloat16, so we keep this in float32
        # and handle dtype conversion in forward()
        self.gru_hidden = int(dim * gru_expansion)
        self._gru_float32 = nn.GRU(
            input_size=dim,
            hidden_size=self.gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # Register as buffer to exclude from dtype conversion but keep in state_dict
        self._gru_float32 = self._gru_float32.float()
        # Project GRU output back to dim if expansion != 1
        if gru_expansion != 1.0:
            self.gru_proj = nn.Linear(self.gru_hidden, dim, bias=False)
        else:
            self.gru_proj = nn.Identity()

        # === Learned mixing weights ===
        # Start with equal weight, let training decide
        self.mix_mamba = nn.Parameter(torch.ones(1) * 0.5)
        self.mix_gru = nn.Parameter(torch.ones(1) * 0.5)

        # === Optional FFN ===
        self.ff_mult = ff_mult
        if ff_mult > 0:
            self.ff_norm = RMSNorm(dim)
            ff_dim = int(dim * ff_mult)
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            )

        self._init_weights()

    def _init_weights(self):
        # GRU projection - small init for residual
        if not isinstance(self.gru_proj, nn.Identity):
            nn.init.normal_(self.gru_proj.weight, std=0.02)

        # FFN - zero init last layer for residual
        if self.ff_mult > 0:
            nn.init.zeros_(self.ff[2].weight)

    def _apply(self, fn):
        """Override _apply to keep GRU in float32 during dtype conversions."""
        # Apply to all submodules including GRU
        for name, module in self.named_children():
            module._apply(fn)

        # After applying, force GRU back to float32 if it was converted
        if self._gru_float32.weight_ih_l0.dtype != torch.float32:
            self._gru_float32 = self._gru_float32.float()

        # Apply to our own parameters (but not submodules)
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    param_data = fn(param.data)
                if param.grad is not None:
                    with torch.no_grad():
                        param.grad.data = fn(param.grad.data)
                if isinstance(param_data, torch.Tensor):
                    param.data = param_data

        # Apply to buffers
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def forward(self, x):
        """
        Args:
            x: [B, T, D] input
        Returns:
            output: [B, T, D]
        """
        # Pre-norm
        x_norm = self.norm(x)

        # === Parallel paths ===
        # Mamba2 path
        h_mamba = self.mamba2(x_norm)

        # GRU path (no hidden state passed - fresh each chunk)
        # cuDNN GRU stays in float32 via _apply() override, input must match
        x_for_gru = x_norm.contiguous().float()
        with torch.amp.autocast('cuda', enabled=False):
            h_gru, _ = self._gru_float32(x_for_gru)
        h_gru = h_gru.to(x_norm.dtype)
        h_gru = self.gru_proj(h_gru)

        # === Combine with learned mixing ===
        # Softmax over mix weights to ensure they sum to ~1
        weights = F.softmax(torch.stack([self.mix_mamba, self.mix_gru]), dim=0)
        h_combined = weights[0] * h_mamba + weights[1] * h_gru

        # Residual
        out = x + h_combined

        # Optional FFN
        if self.ff_mult > 0:
            out = out + self.ff(self.ff_norm(out))

        return out


class HybridMamba2GRULM(nn.Module):
    """
    Hybrid Mamba2 + cuDNN GRU Language Model.

    Processes sequences in parallel (no TBPTT) like Mamba2.
    Each layer has both Mamba2 and GRU paths running in parallel.

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension (must be divisible by mamba_headdim)
        depth: Number of hybrid blocks
        mamba_d_state: SSM state dimension (default 64)
        mamba_expand: Mamba expansion factor (default 2)
        mamba_headdim: Head dimension for SSD (default 64)
        gru_expansion: GRU hidden size multiplier (default 1.0)
        ff_mult: FFN expansion (default 4.0, 0 = no FFN)
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        gru_expansion: float = 1.0,
        ff_mult: float = 0.0,  # No FFN to match Mamba2
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        if not MAMBA2_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Run: pip install mamba-ssm")

        self.dim = dim
        self.depth = depth

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Hybrid blocks
        self.layers = nn.ModuleList([
            HybridMamba2GRUBlock(
                dim=dim,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                gru_expansion=gru_expansion,
                ff_mult=ff_mult,
                layer_idx=i,
            )
            for i in range(depth)
        ])

        # Output
        self.norm_f = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

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

        Args:
            x: [B, T] input token ids
            prev_hidden: Ignored (no TBPTT)
            prev_hiddens: Ignored (no TBPTT)
            prev_conv_buffers: Ignored
            return_next_prev_hidden: Legacy compatibility
            return_prev_hiddens: Legacy compatibility
            return_loss: If True, compute and return loss
            doc_boundaries: Ignored
            **kwargs: Catch extra args

        Returns:
            loss scalar, or (loss, (None, None)) if return_prev_hiddens
        """
        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Hybrid layers
        for layer in self.layers:
            h = layer(h)

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

        # Return format matching train.py interface
        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # No TBPTT, so hiddens are None
        return loss, (None, None)

    def get_mix_weights(self):
        """Return the learned mixing weights for analysis."""
        weights = []
        for i, layer in enumerate(self.layers):
            w = F.softmax(torch.stack([layer.mix_mamba, layer.mix_gru]), dim=0)
            weights.append({
                'layer': i,
                'mamba': w[0].item(),
                'gru': w[1].item(),
            })
        return weights


def count_parameters(model):
    """Count and breakdown parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if 'token_emb' in n)
    mamba = sum(p.numel() for n, p in model.named_parameters() if 'mamba2' in n)
    gru = sum(p.numel() for n, p in model.named_parameters() if 'gru' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff.' in n)
    norm = sum(p.numel() for n, p in model.named_parameters() if 'norm' in n)

    return {
        'total': total,
        'embedding': embedding,
        'mamba2': mamba,
        'gru': gru,
        'ff': ff,
        'norm': norm,
    }


if __name__ == "__main__":
    if not MAMBA2_AVAILABLE:
        print("mamba-ssm not installed, skipping tests")
        exit(0)

    print("Testing HybridMamba2GRULM...")

    # Test different configurations to find ~1B params (no FF, matching Mamba2)
    print("Finding depth for ~1B params (no FF, like Mamba2):")
    for depth in [20, 24, 28, 32, 35, 38]:
        model = HybridMamba2GRULM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            mamba_d_state=64,
            mamba_expand=2,
            mamba_headdim=64,
            gru_expansion=1.0,
            ff_mult=0.0,  # No FF to match Mamba2
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  Mamba2: {counts['mamba2']:,} ({counts['mamba2']/1e6:.1f}M)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")
        del model

    print("\n" + "="*60)
    print("Testing forward/backward pass...")

    model = HybridMamba2GRULM(
        num_tokens=50280,
        dim=2048,
        depth=24,  # Will adjust based on param count
        mamba_d_state=64,
        mamba_expand=2,
        mamba_headdim=64,
        gru_expansion=1.0,
        ff_mult=0.0,  # No FF
    ).cuda().bfloat16()

    counts = count_parameters(model)
    print(f"Model: {counts['total']:,} params ({counts['total']/1e9:.2f}B)")

    x = torch.randint(0, 50280, (2, 256), device='cuda')

    loss = model(x)
    print(f"Forward pass: loss={loss.item():.4f}")

    loss.backward()
    print("Backward pass succeeded!")

    # Check mixing weights
    print("\nMixing weights by layer:")
    for w in model.get_mix_weights():
        print(f"  Layer {w['layer']}: mamba={w['mamba']:.3f}, gru={w['gru']:.3f}")

    print("\nDone!")

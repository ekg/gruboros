"""
Mamba2 + 3-layer FFN: Based on arXiv:2505.06633 findings.

Key insight: 3-layer FFNs with fewer blocks outperform 2-layer FFNs.
- Standard 2-layer: d → 4d → d with GELU
- Our 3-layer: d → 4d → 4d → d with GELU between each layer

The 3-layer FFN adds more nonlinearity (two GELU applications vs one)
which may help with the "injectivity" issue that linear SSMs have.

Paper result: 10 blocks with 3-layer FFN matched 24 blocks with 2-layer FFN
in parameter count (314M vs 323M) but achieved lower loss.
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
    print("Warning: mamba-ssm not installed. Mamba2FFN3 model unavailable.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ThreeLayerFFN(nn.Module):
    """
    3-layer FFN as described in arXiv:2505.06633.

    Structure: d → 4d → 4d → d with GELU nonlinearity between layers.

    This adds more nonlinear transformations compared to standard 2-layer FFN,
    which may help models learn more complex token interactions.
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion

        # Layer 1: d → 4d
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        # Layer 2: 4d → 4d (the extra hidden layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Layer 3: 4d → d
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Standard initialization
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        # Small init for residual path
        nn.init.normal_(self.fc3.weight, std=0.02 / math.sqrt(2))

    def forward(self, x):
        # d → 4d with GELU
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        # 4d → 4d with GELU (the additional nonlinearity)
        h = F.gelu(self.fc2(h))
        h = self.dropout(h)
        # 4d → d
        h = self.fc3(h)
        return h


class Mamba2FFN3Block(nn.Module):
    """
    Single block with Mamba2 + 3-layer FFN.

    Architecture per block:
        h = x + Mamba2(norm1(x))       # Sequence mixing
        out = h + FFN3(norm2(h))       # Token-level nonlinear transform

    The 3-layer FFN provides two GELU nonlinearities per block,
    compared to one in the standard transformer FFN.
    """

    def __init__(
        self,
        dim: int,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        ff_expansion: int = 4,
        dropout: float = 0.0,
        layer_idx: int = None,
    ):
        super().__init__()
        self.dim = dim

        # Pre-norm for Mamba2
        self.norm1 = RMSNorm(dim)

        # Mamba2 for sequence mixing (linear SSM)
        self.mamba2 = Mamba2(
            d_model=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            layer_idx=layer_idx,
        )

        # Pre-norm for FFN
        self.norm2 = RMSNorm(dim)

        # 3-layer FFN for nonlinear token transformations
        self.ffn = ThreeLayerFFN(dim, expansion=ff_expansion, dropout=dropout)

    def forward(self, x):
        # Mamba2 path with residual
        h = x + self.mamba2(self.norm1(x))
        # FFN path with residual
        out = h + self.ffn(self.norm2(h))
        return out


class Mamba2FFN3LM(nn.Module):
    """
    Mamba2 + 3-layer FFN Language Model.

    Based on arXiv:2505.06633: "Attention Is Not All You Need"

    Key design choices:
    - Mamba2 SSD for efficient sequence mixing (linear complexity)
    - 3-layer FFN for stronger nonlinear transformations
    - Uses GELU activation between all FFN layers
    - Fewer blocks needed for same performance (paper finding)

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension (must be divisible by mamba_headdim)
        depth: Number of Mamba2+FFN3 blocks
        mamba_d_state: SSM state dimension (default 64)
        mamba_expand: Mamba expansion factor (default 2)
        mamba_headdim: Head dimension for SSD (default 64)
        ff_expansion: FFN expansion factor (default 4, so 4d hidden dim)
        dropout: Dropout rate
        tie_weights: Whether to tie embedding and output weights
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
        ff_expansion: int = 4,
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

        # Mamba2 + FFN3 blocks
        self.layers = nn.ModuleList([
            Mamba2FFN3Block(
                dim=dim,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                ff_expansion=ff_expansion,
                dropout=dropout,
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
            (other args for API compatibility, mostly ignored)

        Returns:
            loss scalar, or (loss, (None, None)) if return_prev_hiddens
        """
        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Mamba2 + FFN3 layers
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

        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # No TBPTT state
        return loss, (None, None)


def count_parameters(model):
    """Count and breakdown parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if 'token_emb' in n)
    mamba = sum(p.numel() for n, p in model.named_parameters() if 'mamba2' in n)
    ffn = sum(p.numel() for n, p in model.named_parameters() if 'ffn' in n or 'fc' in n)
    norm = sum(p.numel() for n, p in model.named_parameters() if 'norm' in n)

    return {
        'total': total,
        'embedding': embedding,
        'mamba2': mamba,
        'ffn': ffn,
        'norm': norm,
    }


if __name__ == "__main__":
    if not MAMBA2_AVAILABLE:
        print("mamba-ssm not installed, skipping tests")
        exit(0)

    print("Testing Mamba2FFN3LM...")
    print("="*60)

    # Find ~1B params configuration
    print("\nFinding depth for ~1B params with 3-layer FFN:")
    for depth in [12, 14, 16, 18, 20]:
        model = Mamba2FFN3LM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            mamba_d_state=64,
            mamba_expand=2,
            mamba_headdim=64,
            ff_expansion=4,  # d → 4d → 4d → d
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  Mamba2: {counts['mamba2']:,} ({counts['mamba2']/1e6:.1f}M)")
        print(f"  FFN: {counts['ffn']:,} ({counts['ffn']/1e6:.1f}M)")

        # Per-block breakdown
        per_block_mamba = counts['mamba2'] // depth
        per_block_ffn = counts['ffn'] // depth
        print(f"  Per-block: Mamba={per_block_mamba/1e6:.1f}M, FFN={per_block_ffn/1e6:.1f}M")

        del model

    print("\n" + "="*60)
    print("Testing forward/backward pass...")

    model = Mamba2FFN3LM(
        num_tokens=50280,
        dim=2048,
        depth=16,  # Fewer blocks needed with 3-layer FFN
        mamba_d_state=64,
        mamba_expand=2,
        mamba_headdim=64,
        ff_expansion=4,
    ).cuda().bfloat16()

    counts = count_parameters(model)
    print(f"Model: {counts['total']:,} params ({counts['total']/1e9:.2f}B)")

    x = torch.randint(0, 50280, (2, 256), device='cuda')

    loss = model(x)
    print(f"Forward pass: loss={loss.item():.4f}")

    loss.backward()
    print("Backward pass succeeded!")

    print("\nDone!")

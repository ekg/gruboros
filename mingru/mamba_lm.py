"""
Mamba and Mamba2 Language Model wrappers.

These wrap the official mamba-ssm implementations to match our training interface.
Unlike GRU/EMA variants, Mamba processes sequences in parallel (no TBPTT needed).
Mamba2 uses SSD (State Space Duality) for even faster parallel training.

Note on TBPTT: Mamba's selective scan is designed for full-sequence training.
Hidden state carry-over between chunks would require custom modifications.
For fair comparison, we train with full chunk_size as context (no state carry).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba, Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Mamba/Mamba2 models unavailable.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm and residual."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )

    def forward(self, x, inference_params=None):
        return x + self.mamba(self.norm(x), inference_params=inference_params)


class Mamba2Block(nn.Module):
    """Single Mamba2 block with pre-norm and residual."""

    def __init__(
        self,
        dim: int,
        d_state: int = 64,  # Mamba2 uses larger state
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,  # Mamba2 specific
        layer_idx: int = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mamba2 = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx,
        )

    def forward(self, x, inference_params=None):
        return x + self.mamba2(self.norm(x), inference_params=inference_params)


class MambaLM(nn.Module):
    """
    Mamba Language Model.

    Uses the official Mamba selective state space model for sequence modeling.
    Processes sequences in parallel - no TBPTT/chunking needed.

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension
        depth: Number of Mamba blocks
        d_state: SSM state dimension (default 16)
        d_conv: Conv kernel size (default 4)
        expand: Expansion factor for inner dim (default 2)
        dropout: Dropout rate (applied after embedding)
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Run: pip install mamba-ssm")

        self.dim = dim
        self.depth = depth

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
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
        # Standard init for embeddings
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
        **kwargs,  # Catch any extra arguments (actual_length, etc.)
    ):
        """
        Forward pass matching train.py interface.

        Args:
            x: [B, T] input token ids
            prev_hidden: Ignored (Mamba doesn't do TBPTT)
            prev_hiddens: Ignored (Mamba doesn't do TBPTT)
            prev_conv_buffers: Ignored (Mamba doesn't use conv buffers)
            return_next_prev_hidden: Legacy, if True returns (logits, None)
            return_prev_hiddens: If True with return_loss, include prev_hiddens in dict
            return_loss: If True, compute loss and return dict
            doc_boundaries: Ignored
            **kwargs: Catch extra args (actual_length, etc.)

        Returns:
            If return_loss:
                dict with 'loss', 'logits', 'prev_hiddens', 'prev_conv_buffers'
            elif return_next_prev_hidden or return_prev_hiddens:
                (logits, None) or dict
            else:
                logits: [B, T, V] output logits
        """
        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Mamba layers
        for layer in self.layers:
            h = layer(h)

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token (like minLM)
        # logits: [B, T, V], targets: x shifted by 1
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1),
        )

        # Return format matching minLM interface
        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # Return loss and (hiddens, conv_buffers) - Mamba doesn't use TBPTT so both are None
        return loss, (None, None)


class Mamba2LM(nn.Module):
    """
    Mamba2 Language Model.

    Uses Mamba2 with SSD (State Space Duality) for faster parallel training.
    Requires dim to be divisible by headdim.

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension (must be divisible by headdim)
        depth: Number of Mamba2 blocks
        d_state: SSM state dimension (default 64, larger than Mamba1)
        d_conv: Conv kernel size (default 4)
        expand: Expansion factor for inner dim (default 2)
        headdim: Head dimension for SSD (default 64)
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Run: pip install mamba-ssm")

        self.dim = dim
        self.depth = depth

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Mamba2 blocks
        self.layers = nn.ModuleList([
            Mamba2Block(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
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
        **kwargs,  # Catch any extra arguments (actual_length, etc.)
    ):
        """
        Forward pass matching train.py interface.

        Args:
            x: [B, T] input token ids
            prev_hidden: Ignored (Mamba2 doesn't do TBPTT)
            prev_hiddens: Ignored (Mamba2 doesn't do TBPTT)
            prev_conv_buffers: Ignored (Mamba2 doesn't use conv buffers)
            return_next_prev_hidden: Legacy, if True returns (logits, None)
            return_prev_hiddens: If True with return_loss, include prev_hiddens in dict
            return_loss: If True, compute loss and return dict
            doc_boundaries: Ignored
            **kwargs: Catch extra args (actual_length, etc.)

        Returns:
            If return_loss:
                dict with 'loss', 'logits', 'prev_hiddens', 'prev_conv_buffers'
            elif return_next_prev_hidden or return_prev_hiddens:
                (logits, None) or dict
            else:
                logits: [B, T, V] output logits
        """
        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Mamba2 layers
        for layer in self.layers:
            h = layer(h)

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token (like minLM)
        # logits: [B, T, V], targets: x shifted by 1
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1),
        )

        # Return format matching minLM interface
        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # Return loss and (hiddens, conv_buffers) - Mamba2 doesn't use TBPTT so both are None
        return loss, (None, None)


def create_mamba_lm(
    num_tokens: int,
    dim: int,
    depth: int,
    version: int = 1,  # 1 for Mamba, 2 for Mamba2
    **kwargs
):
    """Factory function to create Mamba or Mamba2 LM."""
    if version == 1:
        return MambaLM(num_tokens=num_tokens, dim=dim, depth=depth, **kwargs)
    elif version == 2:
        return Mamba2LM(num_tokens=num_tokens, dim=dim, depth=depth, **kwargs)
    else:
        raise ValueError(f"Unknown Mamba version: {version}")


if __name__ == "__main__":
    if not MAMBA_AVAILABLE:
        print("mamba-ssm not installed, skipping tests")
        exit(0)

    print("Testing MambaLM...")
    model = MambaLM(
        num_tokens=50281,
        dim=512,
        depth=4,
        d_state=16,
        expand=2,
    ).cuda().bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    x = torch.randint(0, 50281, (2, 256), device='cuda')

    logits = model(x)
    print(f"Output shape: {logits.shape}")

    loss = F.cross_entropy(logits.view(-1, 50281), x.view(-1))
    loss.backward()
    print(f"Backward pass succeeded! Loss: {loss.item():.4f}")

    print("\nTesting Mamba2LM...")
    model2 = Mamba2LM(
        num_tokens=50281,
        dim=512,
        depth=4,
        d_state=64,
        expand=2,
        headdim=64,
    ).cuda().bfloat16()

    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Mamba2 Parameters: {total_params2:,}")

    logits2 = model2(x)
    print(f"Mamba2 Output shape: {logits2.shape}")

    loss2 = F.cross_entropy(logits2.view(-1, 50281), x.view(-1))
    loss2.backward()
    print(f"Mamba2 Backward pass succeeded! Loss: {loss2.item():.4f}")

    print("\nDone!")

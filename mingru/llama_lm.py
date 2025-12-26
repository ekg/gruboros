"""
LLaMA-style Transformer Language Model.

Modern efficient transformer architecture for fair comparison with RNNs/SSMs.
Uses: RoPE position embeddings, SwiGLU FFN, RMSNorm, GQA (optional).

This is a minimal implementation focused on training efficiency at 1-2B scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with x."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional GQA."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,  # For GQA
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat k/v heads if using GQA
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim)
            xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim)
            xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        # Attention
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=mask is None,  # Auto causal if no mask provided
            )
        else:
            # Manual attention
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            else:
                # Causal mask
                causal_mask = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), device=x.device),
                    diagonal=1
                )
                scores = scores + causal_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN (more efficient than standard ReLU FFN)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        # SwiGLU uses 2/3 the hidden dim for same param count
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, dropout)
        self.feed_forward = FeedForward(
            dim,
            int(dim * ff_mult),
            dropout=dropout,
        )
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LLaMALM(nn.Module):
    """
    LLaMA-style Transformer Language Model.

    Args:
        num_tokens: Vocabulary size
        dim: Model dimension
        depth: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads (for GQA, None = standard MHA)
        ff_mult: FFN hidden dim multiplier
        max_seq_len: Maximum sequence length for RoPE
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        ff_mult: float = 4.0,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm_f = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for layer in self.layers:
            # Standard initialization
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.normal_(param, std=0.02)

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
        actual_length=None,
        **kwargs,
    ):
        """
        Forward pass matching train.py interface.

        Note: Transformer doesn't use hidden state carry-over (prev_hidden, etc.)
        These args are accepted for compatibility but ignored.
        """
        B, T = x.shape

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()

        if actual_length is not None:
            mask = torch.arange(T - 1, device=x.device).unsqueeze(0) < (actual_length.unsqueeze(1) - 1)
            targets_masked = targets.clone()
            targets_masked[~mask] = -100
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                targets_masked.view(-1),
                ignore_index=-100,
            )
        else:
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                targets.view(-1),
            )

        # Return format matching minLM interface
        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # Return loss and (hiddens, conv_buffers) - Transformer doesn't use TBPTT
        return loss, (None, None)


def get_llama_config(params: str = "1.3b"):
    """Get standard LLaMA configurations at different sizes."""
    configs = {
        "350m": {"dim": 1024, "depth": 24, "n_heads": 16, "ff_mult": 4.0},
        "700m": {"dim": 1536, "depth": 24, "n_heads": 24, "ff_mult": 4.0},
        "1.3b": {"dim": 2048, "depth": 24, "n_heads": 32, "ff_mult": 4.0},
        "1.5b": {"dim": 2048, "depth": 28, "n_heads": 32, "ff_mult": 4.0},
        "2b": {"dim": 2048, "depth": 32, "n_heads": 32, "ff_mult": 4.0},
        "7b": {"dim": 4096, "depth": 32, "n_heads": 32, "ff_mult": 2.68},  # SwiGLU ratio
    }
    return configs.get(params, configs["1.3b"])


if __name__ == "__main__":
    print("Testing LLaMALM...")

    # Test standard 1.3B config
    config = get_llama_config("1.3b")
    model = LLaMALM(
        num_tokens=50281,
        **config,
        max_seq_len=2048,
    ).cuda().bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"LLaMA {config['dim']}x{config['depth']}: {total_params/1e9:.2f}B parameters")

    x = torch.randint(0, 50281, (2, 512), device='cuda')

    loss = model(x)
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass succeeded!")

    # Test different sizes
    for size in ["350m", "700m", "1.3b", "1.5b", "2b"]:
        config = get_llama_config(size)
        model = LLaMALM(num_tokens=50281, **config, max_seq_len=512)
        params = sum(p.numel() for p in model.parameters())
        print(f"{size}: {params/1e9:.2f}B params (dim={config['dim']}, depth={config['depth']})")

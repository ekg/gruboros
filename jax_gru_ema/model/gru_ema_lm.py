"""
Full GRU+EMA Language Model.

Similar to xLSTMLMModel but using GRU+EMA blocks.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from .gru_ema_block import (
    GRUEMABlock,
    GRUEMABlockConfig,
    FeedForwardConfig,
    LayerNorm,
)
from .gru_ema_layer import GRUEMALayerConfig


def small_init(dim: int):
    """Small initialization."""
    return nn.initializers.normal(stddev=jnp.sqrt(2.0 / (5 * dim)))


@dataclass
class GRUEMALMConfig:
    """Configuration for GRU+EMA Language Model."""
    vocab_size: int = 50280
    embedding_dim: int = 512
    num_blocks: int = 12
    expansion_factor: float = 1.0
    ff_mult: float = 0.0  # 0 = no FFN, >0 = add FFN
    ema_alpha: float = 0.01
    use_bias: bool = True
    dropout: float = 0.0
    tie_weights: bool = True
    add_post_blocks_norm: bool = True
    dtype: str = "bfloat16"

    # Block config (will be created in __post_init__)
    gru_ema_block: GRUEMABlockConfig = None

    def __post_init__(self):
        # Create GRU+EMA layer config
        gru_ema_layer = GRUEMALayerConfig(
            embedding_dim=self.embedding_dim,
            expansion_factor=self.expansion_factor,
            ema_alpha=self.ema_alpha,
            use_bias=self.use_bias,
            dropout=self.dropout,
            dtype=self.dtype,
        )

        # Create FFN config if ff_mult > 0
        ffn_config = None
        if self.ff_mult > 0:
            ffn_config = FeedForwardConfig(
                embedding_dim=self.embedding_dim,
                ff_mult=self.ff_mult,
                use_bias=self.use_bias,
                dropout=self.dropout,
                dtype=self.dtype,
            )

        # Create block config
        self.gru_ema_block = GRUEMABlockConfig(
            gru_ema=gru_ema_layer,
            feedforward=ffn_config,
            dtype=self.dtype,
            _num_blocks=self.num_blocks,
        )

    @property
    def _dtype(self) -> jnp.dtype:
        return getattr(jnp, self.dtype)


class GRUEMABlockStack(nn.Module):
    """Stack of GRU+EMA blocks."""
    config: GRUEMALMConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Apply stack of GRU+EMA blocks.

        Args:
            x: Input (B, S, embedding_dim)
            train: Training mode

        Returns:
            x: Output (B, S, embedding_dim)
        """
        for block_idx in range(self.config.num_blocks):
            # Create block config with correct block index
            block_config = deepcopy(self.config.gru_ema_block)
            block_config._block_idx = block_idx
            block_config.__post_init__()

            block = GRUEMABlock(config=block_config, name=f"block_{block_idx}")
            x = block(x, train=train)

        return x


class GRUEMALMModel(nn.Module):
    """
    GRU+EMA Language Model.

    Architecture:
        Token IDs -> Embedding -> BlockStack -> LayerNorm -> LM Head -> Logits

    Supports tied embedding weights (embedding = lm_head.T).
    """
    config: GRUEMALMConfig

    @nn.compact
    def __call__(self, input_ids: jax.Array, train: bool = True) -> jax.Array:
        """
        Forward pass of the language model.

        Args:
            input_ids: Token IDs (B, S)
            train: Training mode

        Returns:
            logits: Output logits (B, S, vocab_size)
        """
        dtype = self.config._dtype

        # Token embedding
        embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim,
            embedding_init=small_init(self.config.embedding_dim),
            dtype=dtype,
            name="token_embedding",
        )
        x = embedding(input_ids)

        # Optional embedding dropout
        if self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout, deterministic=not train)(x)

        # GRU+EMA block stack
        x = GRUEMABlockStack(config=self.config, name="block_stack")(x, train=train)

        # Post-blocks layer norm
        if self.config.add_post_blocks_norm:
            x = LayerNorm(dtype=dtype, name="post_norm")(x)

        # LM head (to logits)
        if self.config.tie_weights:
            # Tied weights: use embedding matrix transposed
            # Get embedding weights
            embed_weights = embedding.embedding
            # Compute logits: x @ embed_weights.T
            logits = x @ embed_weights.T
        else:
            logits = nn.Dense(
                features=self.config.vocab_size,
                kernel_init=small_init(self.config.embedding_dim),
                use_bias=False,
                dtype=jnp.float32,  # Always compute logits in float32
                name="lm_head",
            )(x)

        return logits


def create_model(
    vocab_size: int = 50280,
    dim: int = 512,
    depth: int = 12,
    expansion: float = 1.0,
    ff_mult: float = 0.0,
    ema_alpha: float = 0.01,
    dropout: float = 0.0,
    dtype: str = "bfloat16",
) -> GRUEMALMModel:
    """
    Create a GRU+EMA language model.

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension (embedding_dim)
        depth: Number of blocks
        expansion: GRU hidden dimension = dim * expansion
        ff_mult: FFN intermediate dimension = dim * ff_mult (0 = no FFN)
        ema_alpha: EMA decay rate (0.01 = ~70 token half-life)
        dropout: Dropout rate
        dtype: Model dtype ("bfloat16" or "float32")

    Returns:
        GRUEMALMModel instance
    """
    config = GRUEMALMConfig(
        vocab_size=vocab_size,
        embedding_dim=dim,
        num_blocks=depth,
        expansion_factor=expansion,
        ff_mult=ff_mult,
        ema_alpha=ema_alpha,
        dropout=dropout,
        dtype=dtype,
    )
    return GRUEMALMModel(config=config)

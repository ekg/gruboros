"""
GRU+EMA Block with LayerNorm and residual connections.

Similar to xLSTMBlock but using GRU+EMA instead of mLSTM.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from .gru_ema_layer import GRUEMALayer, GRUEMALayerConfig


@dataclass
class FeedForwardConfig:
    """Configuration for feed-forward network."""
    embedding_dim: int = 512
    ff_mult: float = 4.0
    use_bias: bool = True
    dropout: float = 0.0
    dtype: str = "bfloat16"

    _num_blocks: int = 1

    @property
    def _dtype(self) -> jnp.dtype:
        return getattr(jnp, self.dtype)

    @property
    def intermediate_dim(self) -> int:
        return int(self.embedding_dim * self.ff_mult)


class FeedForward(nn.Module):
    """Simple feed-forward network with SwiGLU activation."""
    config: FeedForwardConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        D = self.config.embedding_dim
        H = self.config.intermediate_dim
        dtype = self.config._dtype

        # Up projection with gating (SwiGLU style)
        x_up = nn.Dense(
            features=2 * H,
            dtype=dtype,
            use_bias=self.config.use_bias,
            name="proj_up",
        )(x)

        x1, x2 = jnp.split(x_up, 2, axis=-1)
        x_gated = jax.nn.swish(x1) * x2

        # Down projection
        y = nn.Dense(
            features=D,
            dtype=dtype,
            use_bias=self.config.use_bias,
            name="proj_down",
        )(x_gated)

        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)

        return y


class LayerNorm(nn.Module):
    """RMSNorm-style layer normalization."""
    dtype: jnp.dtype = jnp.bfloat16
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # RMSNorm
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + self.eps)

        # Learnable scale
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],), self.dtype)

        return x_norm * scale


@dataclass
class GRUEMABlockConfig:
    """Configuration for GRU+EMA block."""
    gru_ema: GRUEMALayerConfig = field(default_factory=GRUEMALayerConfig)
    feedforward: FeedForwardConfig | None = None
    dtype: str = "bfloat16"

    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        # Propagate settings to sub-configs
        if self.gru_ema is not None:
            self.gru_ema._num_blocks = self._num_blocks
            self.gru_ema._block_idx = self._block_idx
            self.gru_ema.dtype = self.dtype

        if self.feedforward is not None:
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.dtype = self.dtype

    @property
    def _dtype(self) -> jnp.dtype:
        return getattr(jnp, self.dtype)


class GRUEMABlock(nn.Module):
    """
    GRU+EMA Block with pre-LayerNorm and residual connections.

    Architecture:
        x -> LayerNorm -> GRUEMALayer -> + x (residual)
        (optional) -> LayerNorm -> FeedForward -> + x (residual)
    """
    config: GRUEMABlockConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        """
        Apply GRU+EMA block.

        Args:
            x: Input (B, S, D)
            train: Training mode
            **kwargs: Additional arguments (for compatibility)

        Returns:
            x: Output (B, S, D)
        """
        dtype = self.config._dtype

        # GRU+EMA branch with pre-norm and residual
        gru_norm = LayerNorm(dtype=dtype, name="gru_norm")
        gru_layer = GRUEMALayer(config=self.config.gru_ema, name="gru_ema")

        x_norm = gru_norm(x)
        x_gru = gru_layer(x_norm, train=train)
        x = x + x_gru  # Residual

        # Optional FFN branch
        if self.config.feedforward is not None:
            ffn_norm = LayerNorm(dtype=dtype, name="ffn_norm")
            ffn = FeedForward(config=self.config.feedforward, name="ffn")

            x_norm = ffn_norm(x)
            x_ffn = ffn(x_norm, train=train)
            x = x + x_ffn  # Residual

        return x

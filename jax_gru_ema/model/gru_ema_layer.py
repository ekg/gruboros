"""
GRU+EMA Layer with up/down projections.

Similar to the mLSTMLayer in xlstm-jax but using GRU+EMA instead.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from .gru_ema_cell import GRUCell, GRUCellConfig, parallel_ema


@dataclass
class GRUEMALayerConfig:
    """Configuration for GRU+EMA layer."""
    embedding_dim: int = 512
    expansion_factor: float = 1.0
    ema_alpha: float = 0.01
    use_bias: bool = True
    dropout: float = 0.0
    dtype: str = "bfloat16"

    # Set by parent config
    _num_blocks: int = 1
    _block_idx: int = 0

    @property
    def _dtype(self) -> jnp.dtype:
        return getattr(jnp, self.dtype)

    @property
    def hidden_dim(self) -> int:
        return int(self.embedding_dim * self.expansion_factor)


def wang_init(embedding_dim: int, num_blocks: int):
    """Wang initialization scaled by number of blocks."""
    return nn.initializers.normal(stddev=2.0 / (num_blocks * jnp.sqrt(embedding_dim)))


def small_init(dim: int):
    """Small initialization."""
    return nn.initializers.normal(stddev=jnp.sqrt(2.0 / (5 * dim)))


class GRUEMALayer(nn.Module):
    """
    GRU+EMA Layer with up/down projections.

    Architecture:
        x -> up_proj (dim -> hidden*2) -> split [x_gru, z]
        x_gru -> GRU (sequential via scan) -> h_gru
        x_gru -> EMA (parallel via cumsum) -> ema
        h_combined = h_gru + ema_weight * ema
        h_gated = h_combined * swish(z)
        y = down_proj (hidden -> dim) + dropout
    """
    config: GRUEMALayerConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        """
        Apply GRU+EMA layer.

        Args:
            x: Input (B, S, embedding_dim)
            train: Training mode for dropout

        Returns:
            y: Output (B, S, embedding_dim)
        """
        B, S, D = x.shape
        H = self.config.hidden_dim
        dtype = self.config._dtype

        # Up-projection: dim -> 2 * hidden (for GRU input and gating)
        x_up = nn.Dense(
            features=2 * H,
            dtype=dtype,
            kernel_init=small_init(D),
            use_bias=self.config.use_bias,
            name="proj_up",
        )(x)

        # Split into GRU input and gating branch
        x_gru, z = jnp.split(x_up, 2, axis=-1)

        # GRU branch (sequential)
        gru_config = GRUCellConfig(
            input_dim=H,
            hidden_dim=H,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
        )
        h_gru = GRUCell(config=gru_config, name="gru")(x_gru)

        # EMA branch (parallel) - on same input as GRU
        ema = parallel_ema(x_gru, self.config.ema_alpha)

        # Learnable EMA combination weight
        ema_weight = self.param(
            'ema_weight',
            nn.initializers.ones,
            (H,),
            dtype,
        )

        # Combine GRU and EMA
        h_combined = h_gru + ema_weight * ema

        # Gating with z branch (like in mLSTM and GLU variants)
        h_gated = h_combined * jax.nn.swish(z)

        # Down-projection: hidden -> dim
        y = nn.Dense(
            features=D,
            dtype=dtype,
            kernel_init=wang_init(D, self.config._num_blocks),
            use_bias=self.config.use_bias,
            name="proj_down",
        )(h_gated)

        # Dropout
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)

        return y

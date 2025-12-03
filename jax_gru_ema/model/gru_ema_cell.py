"""
GRU Cell and parallel EMA implementation in JAX.

This module provides:
- GRUCell: Full non-associative GRU using jax.lax.scan
- parallel_ema: O(S) parallel EMA computation via cumsum trick
"""

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax


def gru_step(
    h: jax.Array,
    x_t: jax.Array,
    Wz: jax.Array,
    Wr: jax.Array,
    Wh: jax.Array,
    Uz: jax.Array,
    Ur: jax.Array,
    Uh: jax.Array,
    bz: jax.Array,
    br: jax.Array,
    bh: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Single GRU step - full non-associative.

    Args:
        h: Hidden state (B, H)
        x_t: Input at time t (B, D)
        W*: Input weights (D, H)
        U*: Recurrent weights (H, H)
        b*: Biases (H,)

    Returns:
        (h_new, h_new): New hidden state (returned twice for scan)
    """
    # Update gate: z = sigmoid(Wz @ x + Uz @ h + bz)
    z = jax.nn.sigmoid(x_t @ Wz + h @ Uz + bz)

    # Reset gate: r = sigmoid(Wr @ x + Ur @ h + br)
    r = jax.nn.sigmoid(x_t @ Wr + h @ Ur + br)

    # Candidate: h_candidate = tanh(Wh @ x + Uh @ (r * h) + bh)
    # This is the non-associative part: candidate depends on r * h
    h_candidate = jnp.tanh(x_t @ Wh + (r * h) @ Uh + bh)

    # New hidden state: h_new = (1 - z) * h + z * h_candidate
    h_new = (1 - z) * h + z * h_candidate

    return h_new, h_new


def gru_sequence(
    x: jax.Array,
    h0: jax.Array,
    Wz: jax.Array,
    Wr: jax.Array,
    Wh: jax.Array,
    Uz: jax.Array,
    Ur: jax.Array,
    Uh: jax.Array,
    bz: jax.Array,
    br: jax.Array,
    bh: jax.Array,
) -> jax.Array:
    """
    Run GRU over a sequence using jax.lax.scan.

    Args:
        x: Input sequence (B, S, D)
        h0: Initial hidden state (B, H)
        W*, U*, b*: GRU parameters

    Returns:
        h_seq: Hidden states for all timesteps (B, S, H)
    """
    B, S, D = x.shape

    # Partial function with fixed weights
    step_fn = partial(
        gru_step,
        Wz=Wz, Wr=Wr, Wh=Wh,
        Uz=Uz, Ur=Ur, Uh=Uh,
        bz=bz, br=br, bh=bh,
    )

    # Transpose to (S, B, D) for scan over time
    x_seq = x.transpose(1, 0, 2)

    # Scan over sequence
    def scan_fn(h, x_t):
        h_new, _ = step_fn(h, x_t)
        return h_new, h_new

    _, h_seq = lax.scan(scan_fn, h0, x_seq)

    # Transpose back to (B, S, H)
    return h_seq.transpose(1, 0, 2)


def parallel_ema(x: jax.Array, alpha: float) -> jax.Array:
    """
    Parallel EMA computation via cumsum trick.

    EMA recurrence: ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}

    Closed form solution:
        ema_t = alpha * sum_{i=0}^{t} (1-alpha)^{t-i} * x_i

    We compute this in O(S) via cumsum:
        1. Weight x by inverse decay: x_weighted[t] = x[t] / decay^t
        2. Cumsum: cumsum[t] = sum_{i=0}^{t} x_weighted[i]
        3. Apply forward decay: ema[t] = alpha * decay^t * cumsum[t]

    Args:
        x: Input sequence (B, S, D)
        alpha: EMA decay rate (0 < alpha < 1). Smaller = longer memory.
               alpha=0.01 gives ~70 token half-life.

    Returns:
        ema: EMA output (B, S, D)
    """
    B, S, D = x.shape
    decay = 1.0 - alpha

    # Time indices (use float32 for numerical stability in powers)
    t = jnp.arange(S, dtype=jnp.float32)

    # Compute decay powers with numerical stability
    # For long sequences, decay^t can underflow. Use log-space.
    log_decay = jnp.log(decay)
    log_decay_powers = log_decay * t  # log(decay^t)

    # For the cumsum trick, we need:
    # x_weighted[t] = x[t] * decay^(-t) = x[t] * exp(-log_decay * t)
    # Then cumsum, then multiply by decay^t * alpha

    # Inverse decay powers for weighting
    inv_log_decay_powers = -log_decay_powers  # log(decay^(-t))

    # Weight input by inverse decay (in log space for stability)
    # x_weighted = x * exp(inv_log_decay_powers)
    x_weighted = x * jnp.exp(inv_log_decay_powers)[None, :, None]

    # Cumulative sum
    x_cumsum = jnp.cumsum(x_weighted, axis=1)

    # Apply forward decay and scale by alpha
    # ema = alpha * cumsum * exp(log_decay_powers)
    ema = alpha * x_cumsum * jnp.exp(log_decay_powers)[None, :, None]

    return ema.astype(x.dtype)


@dataclass
class GRUCellConfig:
    """Configuration for GRU cell."""
    input_dim: int
    hidden_dim: int
    use_bias: bool = True
    dtype: str = "bfloat16"

    @property
    def _dtype(self) -> jnp.dtype:
        return getattr(jnp, self.dtype)


class GRUCell(nn.Module):
    """
    GRU cell implemented as a Flax module.

    Uses jax.lax.scan for efficient sequential processing.
    This is the full non-associative GRU where the reset gate
    modulates the previous hidden state.
    """
    config: GRUCellConfig

    @nn.compact
    def __call__(self, x: jax.Array, h0: jax.Array | None = None) -> jax.Array:
        """
        Run GRU over input sequence.

        Args:
            x: Input sequence (B, S, input_dim)
            h0: Initial hidden state (B, hidden_dim). If None, uses zeros.

        Returns:
            h_seq: Hidden states (B, S, hidden_dim)
        """
        B, S, D = x.shape
        H = self.config.hidden_dim
        dtype = self.config._dtype

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = jnp.zeros((B, H), dtype=dtype)

        # Input weights (D -> H)
        Wz = self.param('Wz', nn.initializers.glorot_uniform(), (D, H), dtype)
        Wr = self.param('Wr', nn.initializers.glorot_uniform(), (D, H), dtype)
        Wh = self.param('Wh', nn.initializers.glorot_uniform(), (D, H), dtype)

        # Recurrent weights (H -> H) - use orthogonal for stability
        # Note: orthogonal init doesn't support bf16 in cuBLAS, so init in float32 then cast
        def orthogonal_init_float32(key, shape, dtype):
            """Initialize orthogonal in float32, then cast to target dtype."""
            w = nn.initializers.orthogonal()(key, shape, jnp.float32)
            return w.astype(dtype)

        Uz = self.param('Uz', orthogonal_init_float32, (H, H), dtype)
        Ur = self.param('Ur', orthogonal_init_float32, (H, H), dtype)
        Uh = self.param('Uh', orthogonal_init_float32, (H, H), dtype)

        # Biases
        if self.config.use_bias:
            bz = self.param('bz', nn.initializers.zeros, (H,), dtype)
            br = self.param('br', nn.initializers.zeros, (H,), dtype)
            bh = self.param('bh', nn.initializers.zeros, (H,), dtype)
        else:
            bz = jnp.zeros((H,), dtype=dtype)
            br = jnp.zeros((H,), dtype=dtype)
            bh = jnp.zeros((H,), dtype=dtype)

        # Run GRU sequence
        h_seq = gru_sequence(
            x, h0,
            Wz, Wr, Wh,
            Uz, Ur, Uh,
            bz, br, bh,
        )

        return h_seq


class GRUEMACore(nn.Module):
    """
    Combined GRU + EMA core computation.

    Runs GRU sequentially (via scan) and EMA in parallel (via cumsum),
    then combines outputs with a learnable weight.
    """
    hidden_dim: int
    ema_alpha: float = 0.01
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply GRU + EMA to input.

        Args:
            x: Input (B, S, D)

        Returns:
            output: Combined GRU + EMA output (B, S, hidden_dim)
        """
        B, S, D = x.shape
        H = self.hidden_dim

        # GRU cell
        gru_config = GRUCellConfig(
            input_dim=D,
            hidden_dim=H,
            use_bias=self.use_bias,
            dtype=str(self.dtype).split('.')[-1],
        )
        h_gru = GRUCell(config=gru_config, name="gru")(x)

        # EMA branch - project input first
        x_ema = nn.Dense(H, dtype=self.dtype, use_bias=self.use_bias, name="ema_proj")(x)
        ema = parallel_ema(x_ema, self.ema_alpha)

        # Learnable combination weight
        ema_weight = self.param(
            'ema_weight',
            nn.initializers.ones,
            (H,),
            self.dtype,
        )

        # Combine: h_gru + ema_weight * ema
        output = h_gru + ema_weight * ema

        return output

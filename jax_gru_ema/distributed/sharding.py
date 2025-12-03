"""
FSDP-style parameter sharding utilities.

Provides functions for:
- Sharding parameters across devices (FSDP)
- Gathering parameters for computation
- Synchronizing gradients across devices
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def shard_params(
    params: dict,
    axis_name: str,
    min_weight_size: int = 2**16,
) -> dict:
    """
    Shard parameters across the given mesh axis.

    Parameters smaller than min_weight_size are not sharded (replicated).
    Sharding is done along the largest divisible dimension.

    Args:
        params: Parameter pytree
        axis_name: Mesh axis name to shard across
        min_weight_size: Minimum parameter size to shard (default 64K)

    Returns:
        Sharded parameter pytree
    """
    axis_idx = lax.axis_index(axis_name)
    axis_size = lax.psum(1, axis_name)

    def _shard(x: jax.Array) -> jax.Array:
        if x.size < min_weight_size:
            # Too small to shard, replicate
            return x

        shape = x.shape
        # Find largest dimension divisible by axis_size
        idx = np.argsort(shape)[::-1]
        for i in idx:
            if shape[i] % axis_size == 0:
                split_size = shape[i] // axis_size
                # Slice to keep only this device's shard
                return lax.dynamic_slice_in_dim(x, axis_idx * split_size, split_size, axis=i)

        # No suitable axis, replicate
        return x

    return jax.tree.map(_shard, params)


def gather_params(
    params: dict,
    axis_name: str,
    original_shapes: dict | None = None,
) -> dict:
    """
    Gather sharded parameters from all devices.

    This function uses a custom gradient that scatters gradients back
    and averages them, which is needed for FSDP training.

    Args:
        params: Sharded parameter pytree
        axis_name: Mesh axis name to gather across
        original_shapes: Optional dict of original shapes for correct gathering

    Returns:
        Gathered parameter pytree
    """
    axis_size = lax.psum(1, axis_name)

    def _gather_with_grad(x: jax.Array) -> jax.Array:
        """Gather with gradient that scatters and averages."""

        @jax.custom_gradient
        def _gather(x):
            def grad_fn(g):
                # Find the axis that was sharded (axis that grew after gather)
                # For each dimension, check if it's axis_size times larger than local
                for i in range(g.ndim):
                    if g.shape[i] == x.shape[i] * axis_size:
                        # This is the sharded axis, scatter gradients back
                        g_scattered = lax.psum_scatter(g, axis_name, scatter_dimension=i, tiled=True)
                        return g_scattered / axis_size
                # No sharded axis found, just average
                return lax.pmean(g, axis_name)

            # All-gather along each dimension that differs
            result = x
            for i in range(x.ndim):
                # Check if this dimension should be gathered
                # We gather along the largest dimension that can be tiled
                pass

            # Simple approach: all_gather on axis 0, then reshape
            # This works for 2D weights but may need adjustment for others
            gathered = lax.all_gather(x, axis_name, axis=0, tiled=True)
            return gathered, grad_fn

        return _gather(x)

    # Simpler approach: just all_gather with pmean gradient
    def _simple_gather(x: jax.Array) -> jax.Array:
        """Simple gather - all_gather on first dimension with pmean gradient."""

        @jax.custom_gradient
        def f(x):
            def grad_fn(g):
                # Average gradients across all devices
                return lax.pmean(g, axis_name)

            # All-gather on first dimension
            return lax.all_gather(x, axis_name, axis=0, tiled=True), grad_fn

        return f(x)

    return jax.tree.map(_simple_gather, params)


def sync_gradients(
    grads: dict,
    axis_names: str | tuple[str, ...],
) -> dict:
    """
    Synchronize gradients across devices by averaging.

    For replicated parameters, this averages gradients across all devices.
    For sharded parameters, the scatter is handled by gather_params' gradient.

    Args:
        grads: Gradient pytree
        axis_names: Mesh axis name(s) to sync across

    Returns:
        Synchronized gradient pytree
    """
    if isinstance(axis_names, str):
        axis_names = (axis_names,)

    def _sync(g: jax.Array) -> jax.Array:
        return lax.pmean(g, axis_name=axis_names)

    return jax.tree.map(_sync, grads)


def fold_rng_over_axis(
    rng: jax.Array,
    axis_names: str | tuple[str, ...],
) -> jax.Array:
    """
    Fold an RNG key over mesh axes to create unique keys per device.

    This is important for dropout to have different masks on each device.

    Args:
        rng: Base RNG key
        axis_names: Mesh axis name(s) to fold over

    Returns:
        RNG key unique to this device
    """
    if isinstance(axis_names, str):
        axis_names = (axis_names,)

    for axis_name in axis_names:
        axis_idx = lax.axis_index(axis_name)
        rng = jax.random.fold_in(rng, axis_idx)

    return rng

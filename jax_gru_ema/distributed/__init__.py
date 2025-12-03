"""
Distributed training utilities for JAX GRU+EMA.

Provides FSDP-style parameter sharding across multiple GPUs.
"""

from .mesh import create_mesh, get_mesh_axis_names
from .sharding import (
    shard_params,
    gather_params,
    sync_gradients,
    fold_rng_over_axis,
)

__all__ = [
    "create_mesh",
    "get_mesh_axis_names",
    "shard_params",
    "gather_params",
    "sync_gradients",
    "fold_rng_over_axis",
]

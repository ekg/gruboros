"""
Device mesh utilities for distributed training.

Simplified FSDP mesh setup for single-node multi-GPU training.
"""

import logging

import jax
import numpy as np
from jax.sharding import Mesh

LOGGER = logging.getLogger(__name__)


def create_mesh(
    fsdp_axis_size: int = -1,
    data_axis_size: int = 1,
    fsdp_axis_name: str = "fsdp",
    data_axis_name: str = "data",
) -> Mesh:
    """
    Create a device mesh for FSDP training.

    For simple FSDP (all GPUs shard params), use:
        create_mesh(fsdp_axis_size=-1, data_axis_size=1)

    For data parallel (replicate params), use:
        create_mesh(fsdp_axis_size=1, data_axis_size=-1)

    For hybrid (e.g., 2 replicas × 4 FSDP shards), use:
        create_mesh(fsdp_axis_size=4, data_axis_size=2)

    Args:
        fsdp_axis_size: Number of devices for FSDP axis. -1 = auto (all devices / data_axis_size)
        data_axis_size: Number of devices for data axis. -1 = auto (all devices / fsdp_axis_size)
        fsdp_axis_name: Name for FSDP axis (default: "fsdp")
        data_axis_name: Name for data axis (default: "data")

    Returns:
        Mesh object for use with jax.sharding
    """
    devices = jax.devices()
    num_devices = len(devices)

    if jax.process_index() == 0:
        LOGGER.info(f"Creating mesh with {num_devices} devices")

    # Compute axis sizes
    if fsdp_axis_size == -1 and data_axis_size == -1:
        # Default: pure FSDP
        fsdp_axis_size = num_devices
        data_axis_size = 1
    elif fsdp_axis_size == -1:
        fsdp_axis_size = num_devices // data_axis_size
    elif data_axis_size == -1:
        data_axis_size = num_devices // fsdp_axis_size

    assert (
        fsdp_axis_size * data_axis_size == num_devices
    ), f"fsdp_axis_size ({fsdp_axis_size}) × data_axis_size ({data_axis_size}) != num_devices ({num_devices})"

    # Reshape devices into 2D array: (data, fsdp)
    device_array = np.array(devices).reshape(data_axis_size, fsdp_axis_size)

    # Create mesh
    mesh = Mesh(device_array, (data_axis_name, fsdp_axis_name))

    if jax.process_index() == 0:
        LOGGER.info(f"Created mesh: {mesh}")

    return mesh


def get_mesh_axis_names(mesh: Mesh) -> tuple[str, str]:
    """
    Get the axis names from a mesh.

    Returns:
        (data_axis_name, fsdp_axis_name)
    """
    return mesh.axis_names

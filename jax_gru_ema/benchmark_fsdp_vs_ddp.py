#!/usr/bin/env python3
"""
Benchmark FSDP vs DDP performance for JAX GRU+EMA.

Compares:
1. FSDP: Parameters sharded across 8 GPUs, each GPU has 1/8 of params
2. DDP: Parameters replicated, batch sharded across 8 GPUs

Key metrics:
- Maximum batch size that fits in memory
- Training throughput (tokens/sec)
- Memory usage per GPU
"""

import time
import sys
sys.path.insert(0, '/home/erikg/gruboros')

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax_gru_ema.model.gru_ema_lm import create_model
from jax_gru_ema.distributed.mesh import create_mesh
from jax_gru_ema.distributed.sharding import sync_gradients, fold_rng_over_axis

import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding


def count_params(params):
    """Count parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


def get_gpu_memory():
    """Get GPU memory usage (approximate via JAX)."""
    # JAX doesn't have direct memory query, but we can check via device_buffer
    return "N/A"


def benchmark_config(
    mode: str,  # "fsdp" or "ddp"
    dim: int,
    depth: int,
    batch_size: int,
    chunk_size: int,
    num_steps: int = 10,
    warmup_steps: int = 3,
):
    """
    Benchmark a specific configuration.

    Args:
        mode: "fsdp" (param sharding) or "ddp" (data parallel)
        dim: Model dimension
        depth: Number of layers
        batch_size: Per-device batch size
        chunk_size: Sequence length
        num_steps: Number of steps to benchmark
        warmup_steps: Warmup steps (not timed)

    Returns:
        Dict with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: mode={mode}, dim={dim}, depth={depth}")
    print(f"batch_size={batch_size}, chunk_size={chunk_size}")
    print(f"{'='*60}")

    # Create model
    model = create_model(
        vocab_size=50280,
        dim=dim,
        depth=depth,
        expansion=1.0,
        ff_mult=0.0,
        dtype="bfloat16",
    )

    # Create mesh based on mode
    num_devices = len(jax.devices())
    if mode == "fsdp":
        # Pure FSDP: shard params, replicate data across batch
        mesh = create_mesh(fsdp_axis_size=num_devices, data_axis_size=1)
    else:  # ddp
        # Pure DDP: replicate params, shard data
        mesh = create_mesh(fsdp_axis_size=1, data_axis_size=num_devices)

    data_axis_name, fsdp_axis_name = mesh.axis_names
    print(f"Mesh: {mesh}")
    print(f"Axis names: data={data_axis_name}, fsdp={fsdp_axis_name}")

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    init_rng, train_rng = jax.random.split(rng)

    dummy_input = jnp.ones((1, chunk_size), dtype=jnp.int32)

    with mesh:
        params = model.init(init_rng, dummy_input, train=False)["params"]

    n_params = count_params(params)
    print(f"Total parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    # Create optimizer
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.01)

    with mesh:
        opt_state = optimizer.init(params)

    # Define training step
    def train_step_inner(params, opt_state, batch, rng, fsdp_axis_name, data_axis_name):
        """Inner training step."""

        def loss_fn(params):
            logits = model.apply(
                {"params": params},
                batch["input_ids"],
                train=True,
                rngs={"dropout": rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["target_ids"]
            )
            return loss.mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Sync gradients
        grads = sync_gradients(grads, (fsdp_axis_name, data_axis_name))
        loss = jax.lax.pmean(loss, axis_name=(fsdp_axis_name, data_axis_name))

        # Update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Create sharded step
    step_fn = partial(
        train_step_inner,
        fsdp_axis_name=fsdp_axis_name,
        data_axis_name=data_axis_name,
    )

    # Batch spec depends on mode
    if mode == "fsdp":
        batch_spec = P()  # Replicated batch for FSDP
    else:  # ddp
        batch_spec = P(data_axis_name)  # Sharded batch for DDP

    sharded_step = shard_map(
        step_fn,
        mesh=mesh,
        in_specs=(P(), P(), batch_spec, P()),  # params, opt_state, batch, rng
        out_specs=(P(), P(), P()),  # params, opt_state, loss
        check_rep=False,
    )

    train_step = jax.jit(sharded_step)

    # Create dummy batch
    total_batch = batch_size * num_devices if mode == "ddp" else batch_size

    def make_batch():
        return {
            "input_ids": jnp.zeros((total_batch, chunk_size), dtype=jnp.int32),
            "target_ids": jnp.zeros((total_batch, chunk_size), dtype=jnp.int32),
        }

    batch = make_batch()

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    try:
        with mesh:
            for i in range(warmup_steps):
                train_rng, step_rng = jax.random.split(train_rng)
                params, opt_state, loss = train_step(params, opt_state, batch, step_rng)
                jax.block_until_ready(loss)
                print(f"  Warmup step {i+1}: loss={float(loss):.4f}")
    except Exception as e:
        print(f"ERROR during warmup: {e}")
        return {
            "mode": mode,
            "dim": dim,
            "depth": depth,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "success": False,
            "error": str(e),
        }

    # Benchmark
    print(f"Benchmarking ({num_steps} steps)...")
    tokens_per_step = total_batch * chunk_size

    start_time = time.time()
    with mesh:
        for i in range(num_steps):
            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, loss = train_step(params, opt_state, batch, step_rng)
        jax.block_until_ready(loss)
    elapsed = time.time() - start_time

    tokens_per_sec = (tokens_per_step * num_steps) / elapsed
    steps_per_sec = num_steps / elapsed

    result = {
        "mode": mode,
        "dim": dim,
        "depth": depth,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "total_batch": total_batch,
        "n_params": n_params,
        "success": True,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "final_loss": float(loss),
    }

    print(f"\nResults:")
    print(f"  Total batch size: {total_batch}")
    print(f"  Elapsed: {elapsed:.2f}s for {num_steps} steps")
    print(f"  Throughput: {steps_per_sec:.2f} steps/sec")
    print(f"  Throughput: {tokens_per_sec/1e6:.2f}M tokens/sec")

    return result


def find_max_batch_size(mode: str, dim: int, depth: int, chunk_size: int):
    """
    Binary search to find maximum batch size.

    Args:
        mode: "fsdp" or "ddp"
        dim: Model dimension
        depth: Number of layers
        chunk_size: Sequence length

    Returns:
        Maximum batch size that works
    """
    print(f"\n{'#'*60}")
    print(f"Finding max batch size for {mode}, dim={dim}, depth={depth}")
    print(f"{'#'*60}")

    # Start with powers of 2
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_working = 0

    for bs in batch_sizes:
        print(f"\nTrying batch_size={bs}...")
        try:
            result = benchmark_config(
                mode=mode,
                dim=dim,
                depth=depth,
                batch_size=bs,
                chunk_size=chunk_size,
                num_steps=3,
                warmup_steps=1,
            )
            if result["success"]:
                max_working = bs
                print(f"  -> SUCCESS with batch_size={bs}")
            else:
                print(f"  -> FAILED: {result.get('error', 'unknown')}")
                break
        except Exception as e:
            print(f"  -> FAILED: {e}")
            break

    return max_working


def main():
    print("JAX GRU+EMA: FSDP vs DDP Benchmark")
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {len(jax.devices())}")
    print()

    # Test configurations
    configs = [
        # Small model (should fit easily)
        {"dim": 512, "depth": 8, "chunk_size": 512},
        # Medium model
        {"dim": 1024, "depth": 12, "chunk_size": 512},
        # Large model (closer to 1B)
        {"dim": 2048, "depth": 20, "chunk_size": 512},
    ]

    results = []

    for cfg in configs:
        dim, depth, chunk_size = cfg["dim"], cfg["depth"], cfg["chunk_size"]

        # Find max batch size for each mode
        for mode in ["fsdp", "ddp"]:
            max_bs = find_max_batch_size(mode, dim, depth, chunk_size)
            print(f"\n*** {mode.upper()}: max_batch_size = {max_bs} ***")

            if max_bs > 0:
                # Run full benchmark at max batch size
                result = benchmark_config(
                    mode=mode,
                    dim=dim,
                    depth=depth,
                    batch_size=max_bs,
                    chunk_size=chunk_size,
                    num_steps=20,
                    warmup_steps=5,
                )
                results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Mode':<6} {'Dim':<6} {'Depth':<6} {'BatchSize':<10} {'Tokens/sec':<15} {'Steps/sec':<10}")
    print("-"*80)
    for r in results:
        if r["success"]:
            print(f"{r['mode']:<6} {r['dim']:<6} {r['depth']:<6} {r['total_batch']:<10} "
                  f"{r['tokens_per_sec']/1e6:.2f}M{'':<8} {r['steps_per_sec']:.2f}")
        else:
            print(f"{r['mode']:<6} {r['dim']:<6} {r['depth']:<6} FAILED: {r.get('error', 'unknown')[:30]}")


if __name__ == "__main__":
    main()

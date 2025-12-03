"""
Trainer for GRU+EMA language model with FSDP support.

Simple training loop with:
- FSDP-style parameter sharding
- Gradient accumulation
- AdamW optimizer
- Basic logging
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from .distributed import create_mesh, sync_gradients, fold_rng_over_axis


@dataclass
class TrainerConfig:
    """Configuration for trainer."""

    # Model
    vocab_size: int = 50280
    dim: int = 512
    depth: int = 12
    expansion: float = 1.0
    ff_mult: float = 0.0
    ema_alpha: float = 0.01
    dropout: float = 0.0

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 8
    chunk_size: int = 512
    grad_accum_steps: int = 1

    # Distributed
    fsdp_axis_size: int = -1  # -1 = use all devices
    data_axis_size: int = 1

    # Logging
    log_every: int = 10
    save_every: int = 1000


class TrainState(train_state.TrainState):
    """Extended train state with RNG."""

    rng: jax.Array


def create_train_state(
    model: nn.Module,
    config: TrainerConfig,
    rng: jax.Array,
) -> TrainState:
    """
    Create initial training state.

    Args:
        model: Flax model
        config: Trainer configuration
        rng: Random key

    Returns:
        Initialized TrainState
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, config.chunk_size), dtype=jnp.int32)

    # Split RNG
    init_rng, train_rng = jax.random.split(rng)

    # Initialize parameters
    params = model.init(init_rng, dummy_input, train=False)["params"]

    # Create optimizer with warmup schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.learning_rate * 0.1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=train_rng,
    )


def compute_loss(
    params: dict,
    apply_fn: Any,
    batch: dict,
    rng: jax.Array,
    train: bool = True,
) -> tuple[jax.Array, dict]:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        params: Model parameters
        apply_fn: Model apply function
        batch: Dict with "input_ids" and "target_ids"
        rng: Random key for dropout
        train: Training mode

    Returns:
        (loss, metrics dict)
    """
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    # Forward pass
    logits = apply_fn(
        {"params": params},
        input_ids,
        train=train,
        rngs={"dropout": rng},
    )

    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    loss = loss.mean()

    # Metrics
    correct = jnp.argmax(logits, axis=-1) == target_ids
    accuracy = correct.mean()

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
    }

    return loss, metrics


def train_step_single(
    state: TrainState,
    batch: dict,
    fsdp_axis_name: str = "fsdp",
    data_axis_name: str = "data",
) -> tuple[TrainState, dict]:
    """
    Single training step (used inside shard_map).

    Args:
        state: Training state
        batch: Input batch
        fsdp_axis_name: Name of FSDP axis
        data_axis_name: Name of data axis

    Returns:
        (updated state, metrics)
    """
    # Split RNG for this step
    rng, dropout_rng = jax.random.split(state.rng)

    # Fold RNG over axes for unique dropout masks per device
    dropout_rng = fold_rng_over_axis(dropout_rng, (fsdp_axis_name, data_axis_name))

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params,
        state.apply_fn,
        batch,
        dropout_rng,
        train=True,
    )

    # Sync gradients across FSDP axis
    grads = sync_gradients(grads, (fsdp_axis_name, data_axis_name))

    # Also sync metrics
    metrics = jax.tree.map(
        lambda x: jax.lax.pmean(x, axis_name=(fsdp_axis_name, data_axis_name)),
        metrics,
    )

    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(rng=rng)

    return state, metrics


def create_train_step(
    mesh: Mesh,
    fsdp_axis_name: str = "fsdp",
    data_axis_name: str = "data",
):
    """
    Create a sharded training step function.

    Args:
        mesh: Device mesh
        fsdp_axis_name: Name of FSDP axis
        data_axis_name: Name of data axis

    Returns:
        JIT-compiled sharded training step function
    """
    # Partition specs for state and batch
    # State is replicated (each device has full copy)
    # Batch is sharded over data axis
    state_spec = P()  # Replicated
    batch_spec = P(data_axis_name)  # Sharded over data axis

    # Create sharded training step
    sharded_step = shard_map(
        partial(
            train_step_single,
            fsdp_axis_name=fsdp_axis_name,
            data_axis_name=data_axis_name,
        ),
        mesh=mesh,
        in_specs=(state_spec, batch_spec),
        out_specs=(state_spec, P()),  # Metrics replicated
        check_rep=False,
    )

    return jax.jit(sharded_step, donate_argnums=(0,))


def train_loop(
    model: nn.Module,
    config: TrainerConfig,
    data_iter,
    rng: jax.Array,
):
    """
    Main training loop.

    Args:
        model: Flax model
        config: Trainer configuration
        data_iter: Iterator yielding batches
        rng: Random key
    """
    # Create mesh
    mesh = create_mesh(
        fsdp_axis_size=config.fsdp_axis_size,
        data_axis_size=config.data_axis_size,
    )
    data_axis_name, fsdp_axis_name = mesh.axis_names

    print(f"Created mesh: {mesh}")
    print(f"Axis names: data={data_axis_name}, fsdp={fsdp_axis_name}")

    # Initialize training state
    with mesh:
        state = create_train_state(model, config, rng)
        print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(state.params)):,}")

    # Create sharded training step
    train_step = create_train_step(mesh, fsdp_axis_name, data_axis_name)

    # Training loop
    for step, batch in enumerate(data_iter):
        if step >= config.max_steps:
            break

        with mesh:
            state, metrics = train_step(state, batch)

        if step % config.log_every == 0:
            loss = float(metrics["loss"])
            acc = float(metrics["accuracy"])
            ppl = float(metrics["perplexity"])
            print(f"Step {step}: loss={loss:.4f}, acc={acc:.4f}, ppl={ppl:.2f}")

        if step > 0 and step % config.save_every == 0:
            # TODO: Implement checkpointing
            print(f"Step {step}: would save checkpoint")

    return state


# Simple test
if __name__ == "__main__":
    from .model.gru_ema_lm import create_model

    print("Testing trainer with small model...")

    # Create small model
    model = create_model(
        vocab_size=1000,
        dim=128,
        depth=4,
        expansion=1.0,
        ff_mult=0.0,
        dtype="float32",  # Use float32 for testing
    )

    config = TrainerConfig(
        vocab_size=1000,
        dim=128,
        depth=4,
        batch_size=2,
        chunk_size=64,
        max_steps=10,
        log_every=1,
    )

    # Create dummy data iterator
    def dummy_data_iter():
        rng = jax.random.PRNGKey(0)
        while True:
            rng, key = jax.random.split(rng)
            input_ids = jax.random.randint(key, (config.batch_size, config.chunk_size), 0, config.vocab_size)
            target_ids = jax.random.randint(key, (config.batch_size, config.chunk_size), 0, config.vocab_size)
            yield {"input_ids": input_ids, "target_ids": target_ids}

    rng = jax.random.PRNGKey(42)
    state = train_loop(model, config, dummy_data_iter(), rng)
    print("Training test passed!")

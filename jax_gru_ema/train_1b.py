#!/usr/bin/env python3
"""
Train 1B parameter GRU+EMA language model in JAX.

Uses the SAME data loading as train.py:
- Raw text file with 0x1e document delimiters
- On-the-fly tiktoken tokenization
- Memory-mapped file access
"""

import argparse
import os
import sys
import time
import random
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
# Force JAX to initialize cuDNN BEFORE importing torch (via mingru)
# This prevents PyTorch's bundled cuDNN 9.1 from conflicting with JAX's 9.8+ requirement
_ = jax.devices()
_ = jax.random.PRNGKey(0)

import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding

from jax_gru_ema.model.gru_ema_lm import create_model
from jax_gru_ema.distributed import create_mesh, sync_gradients, fold_rng_over_axis

# Reuse existing tokenizer infrastructure (imports torch which has older cuDNN)
from mingru.tokenizers import get_tokenizer


def count_params(params):
    """Count parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


class DocumentStreamDataset:
    """
    Document-aware streaming dataset with tiktoken.

    Same semantics as train.py's DocumentStreamDataset:
    - Memory-mapped raw text file
    - 0x1e document delimiters
    - On-the-fly tiktoken tokenization
    - Each rank starts at different position
    """

    def __init__(self, filepath, chunk_size, tokenizer, seed=42, rank=0, world_size=1):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        # Memory-map the file
        self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        self.file_size = len(self.mmap)

        # Each rank gets unique starting position
        rng = random.Random(seed + rank)
        self.position = rng.randint(0, self.file_size - 1)

        # Stats
        self.documents_processed = 0
        self.bytes_processed = 0
        self.tokens_processed = 0
        self.wraps = 0

        # Scan to next document boundary
        self._scan_to_next_document()

        # Buffers
        self.token_buffer = []
        self.text_buffer = b''
        self.read_chunk_size = 32768  # 32KB at a time

        print(f"Rank {rank}: Dataset initialized at position {self.position}")

    def _scan_to_next_document(self):
        """Scan forward to next document boundary."""
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1

        if self.position >= self.file_size:
            self.position = 0
            self.wraps += 1
        else:
            self.position += 1  # Skip delimiter
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1

    def get_next_chunk(self):
        """
        Get next chunk of tokens.

        Returns:
            (tokens, is_end_of_doc, actual_length)
        """
        while len(self.token_buffer) < self.chunk_size:
            # Read bytes
            bytes_to_read = min(self.read_chunk_size, self.file_size - self.position)
            if bytes_to_read == 0:
                self.position = 0
                self.wraps += 1
                bytes_to_read = min(self.read_chunk_size, self.file_size)

            byte_chunk = bytes(self.mmap[self.position:self.position + bytes_to_read])
            self.position += bytes_to_read
            self.bytes_processed += bytes_to_read

            # Check for document boundary
            doc_idx = byte_chunk.find(b'\x1e')

            if doc_idx != -1:
                # Found boundary
                self.text_buffer += byte_chunk[:doc_idx]

                # Tokenize accumulated text
                if len(self.text_buffer) > 0:
                    try:
                        text = self.text_buffer.decode('utf-8', errors='ignore')
                        tokens = self.tokenizer.encode(text)
                        self.token_buffer.extend(tokens)
                        self.text_buffer = b''
                    except Exception as e:
                        print(f"Tokenization error: {e}")
                        self.text_buffer = b''

                self.documents_processed += 1

                # Adjust position past delimiter
                self.position = self.position - len(byte_chunk) + doc_idx + 1
                if self.position >= self.file_size:
                    self.position = 0
                    self.wraps += 1

                # Return partial chunk at doc boundary
                if len(self.token_buffer) > 0:
                    actual_length = min(len(self.token_buffer), self.chunk_size)
                    chunk = np.zeros(self.chunk_size, dtype=np.int32)
                    chunk[:actual_length] = self.token_buffer[:actual_length]
                    self.token_buffer = self.token_buffer[actual_length:]
                    self.tokens_processed += actual_length
                    return chunk, True, actual_length
            else:
                # No boundary, accumulate
                self.text_buffer += byte_chunk

                # Periodically tokenize to avoid huge buffer
                if len(self.text_buffer) >= self.read_chunk_size * 4:
                    try:
                        text = self.text_buffer.decode('utf-8', errors='ignore')
                        tokens = self.tokenizer.encode(text)
                        self.token_buffer.extend(tokens)
                        self.text_buffer = b''
                    except Exception as e:
                        print(f"Tokenization error: {e}")
                        self.text_buffer = b''

        # Return full chunk
        chunk = np.array(self.token_buffer[:self.chunk_size], dtype=np.int32)
        self.token_buffer = self.token_buffer[self.chunk_size:]
        self.tokens_processed += self.chunk_size
        return chunk, False, self.chunk_size


def create_batch_iterator(datasets, batch_size):
    """
    Create batched iterator from multiple per-rank datasets.

    Args:
        datasets: List of DocumentStreamDataset (one per device)
        batch_size: Per-device batch size

    Yields:
        Dict with input_ids and target_ids as JAX arrays
    """
    num_devices = len(datasets)

    while True:
        # Collect batch_size samples from each device
        all_inputs = []
        all_targets = []

        for ds in datasets:
            for _ in range(batch_size):
                chunk, _, _ = ds.get_next_chunk()
                # input_ids = chunk[:-1], target_ids = chunk[1:]
                # But we need full chunk_size, so: input = chunk, target = shifted
                all_inputs.append(chunk[:-1])
                all_targets.append(chunk[1:])

        # Stack: (num_devices * batch_size, chunk_size - 1)
        yield {
            "input_ids": jnp.array(np.stack(all_inputs)),
            "target_ids": jnp.array(np.stack(all_targets)),
        }


def train_step_inner(params, opt_state, batch, rng, optimizer, model, data_axis_name, fsdp_axis_name):
    """Inner training step (runs on each device)."""

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

    # Sync gradients across all axes
    grads = sync_gradients(grads, (fsdp_axis_name, data_axis_name))
    loss = jax.lax.pmean(loss, axis_name=(fsdp_axis_name, data_axis_name))

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def main():
    parser = argparse.ArgumentParser(description="Train 1B JAX GRU+EMA model")
    parser.add_argument("--data", type=str, required=True, help="Path to raw text data (same as train.py)")
    parser.add_argument("--output", type=str, default="output_jax", help="Output directory")

    # Model config (defaults to 1B)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--expansion", type=float, default=1.0)
    parser.add_argument("--ff_mult", type=float, default=0.0)
    parser.add_argument("--ema_alpha", type=float, default=0.01)
    parser.add_argument("--vocab_size", type=int, default=50281)  # tiktoken p50k_base has 50281 tokens

    # Tokenizer (same as train.py)
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["byte", "tiktoken"])
    parser.add_argument("--tiktoken_encoding", type=str, default="p50k_base")

    # Training config
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.033)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Distributed config
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP (param sharding)")
    parser.add_argument("--ddp", action="store_true", help="Use DDP (data parallel)")

    # Logging
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    # Setup devices
    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX devices: {devices}")
    print(f"Number of devices: {num_devices}")

    # Create tokenizer (same as train.py)
    if args.tokenizer == "tiktoken":
        tokenizer = get_tokenizer("tiktoken", encoding_name=args.tiktoken_encoding)
    else:
        tokenizer = get_tokenizer("byte")
    print(f"Tokenizer: {tokenizer}")

    # Create mesh based on mode
    if args.fsdp:
        mesh = create_mesh(fsdp_axis_size=num_devices, data_axis_size=1)
    else:  # DDP (default)
        mesh = create_mesh(fsdp_axis_size=1, data_axis_size=num_devices)

    data_axis_name, fsdp_axis_name = mesh.axis_names
    print(f"Mesh: {mesh}")
    print(f"Axis names: data={data_axis_name}, fsdp={fsdp_axis_name}")

    # Create model
    print(f"\nCreating model: dim={args.dim}, depth={args.depth}")
    model = create_model(
        vocab_size=args.vocab_size,
        dim=args.dim,
        depth=args.depth,
        expansion=args.expansion,
        ff_mult=args.ff_mult,
        ema_alpha=args.ema_alpha,
        dtype="bfloat16",
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    init_rng, train_rng = jax.random.split(rng)

    # Note: chunk_size - 1 because input/target are shifted
    dummy_input = jnp.ones((1, args.chunk_size - 1), dtype=jnp.int32)

    with mesh:
        params = model.init(init_rng, dummy_input, train=False)["params"]

    n_params = count_params(params)
    print(f"Total parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.train_steps,
        end_value=args.lr * 0.1,
    )

    optimizer_chain = [optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)]
    if args.grad_clip > 0:
        optimizer_chain.insert(0, optax.clip_by_global_norm(args.grad_clip))
    optimizer = optax.chain(*optimizer_chain)

    with mesh:
        opt_state = optimizer.init(params)

    # Create sharded training step
    step_fn = partial(
        train_step_inner,
        optimizer=optimizer,
        model=model,
        fsdp_axis_name=fsdp_axis_name,
        data_axis_name=data_axis_name,
    )

    # Batch spec depends on mode
    if args.fsdp:
        batch_spec = P()  # Replicated batch for FSDP
    else:
        batch_spec = P(data_axis_name)  # Sharded batch for DDP

    sharded_step = shard_map(
        step_fn,
        mesh=mesh,
        in_specs=(P(), P(), batch_spec, P()),  # params, opt_state, batch, rng
        out_specs=(P(), P(), P()),  # params, opt_state, loss
        check_rep=False,
    )

    train_step = jax.jit(sharded_step, donate_argnums=(0, 1))

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create data loaders - one per device, just like train.py
    print(f"\nLoading data from: {args.data}")
    datasets = []
    for rank in range(num_devices):
        ds = DocumentStreamDataset(
            filepath=args.data,
            chunk_size=args.chunk_size,
            tokenizer=tokenizer,
            seed=42,
            rank=rank,
            world_size=num_devices,
        )
        datasets.append(ds)

    batch_iter = create_batch_iterator(datasets, args.batch_size)

    # Training loop
    print(f"\nStarting training for {args.train_steps} steps")
    total_batch = args.batch_size * num_devices
    seq_len = args.chunk_size - 1
    print(f"Batch size: {args.batch_size} per device x {num_devices} devices = {total_batch} total")
    print(f"Tokens per step: {total_batch * seq_len:,}")

    step = 0
    total_tokens = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0

    with mesh:
        for batch in batch_iter:
            if step >= args.train_steps:
                break

            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, loss = train_step(params, opt_state, batch, step_rng)

            total_tokens += total_batch * seq_len

            if step % args.log_every == 0:
                jax.block_until_ready(loss)
                now = time.time()
                elapsed = now - start_time
                # Calculate instantaneous rate (excluding step 0 JIT)
                if step > 0:
                    interval = now - last_log_time
                    interval_tokens = (step - last_log_step) * total_batch * seq_len
                    tokens_per_sec = interval_tokens / interval if interval > 0 else 0
                else:
                    tokens_per_sec = 0
                print(f"Step {step}: loss={float(loss):.4f}, "
                      f"tok/s={tokens_per_sec:,.0f}, "
                      f"elapsed={elapsed:.1f}s", flush=True)
                last_log_time = now
                last_log_step = step

            step += 1

    print(f"\nTraining complete!")
    print(f"Final loss: {float(loss):.4f}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

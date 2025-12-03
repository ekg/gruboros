# JAX GRU+EMA: Work in Progress

## Goal

Port GRU+EMA to JAX with xlstm-jax's distributed training infrastructure, keeping gruboros's mmap'd document streaming approach. Get multi-GPU training that actually works (unlike FlashRNN+DDP).

## Architecture Overview

```
GRUEMALMModel
├── Embedding (vocab_size → dim)
├── GRUEMABlockStack (N × GRUEMABlock)
│   └── GRUEMABlock
│       ├── LayerNorm → GRUEMALayer → residual
│       └── (optional) LayerNorm → FFN → residual
├── LayerNorm
└── LM head (dim → vocab_size, tied weights)
```

### GRUEMALayer

```python
class GRUEMALayer(nn.Module):
    """GRU + parallel EMA layer."""
    config: GRUEMALayerConfig

    @nn.compact
    def __call__(self, x, train=True):
        B, S, D = x.shape
        hidden_dim = int(D * self.config.expansion)

        # Up-projection
        x_up = nn.Dense(hidden_dim * 2)(x)  # for GRU gates + candidate
        x_ema = nn.Dense(hidden_dim)(x)      # EMA input

        # GRU branch (sequential via scan)
        h_gru = self.gru_scan(x_up, hidden_dim)

        # EMA branch (parallel via cumsum)
        ema = self.parallel_ema(x_ema)

        # Combine with learnable weight
        ema_weight = self.param('ema_weight', nn.initializers.ones, (hidden_dim,))
        combined = h_gru + ema_weight * ema

        # Down-projection
        y = nn.Dense(D)(combined)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=not train)
        return y
```

---

## Phase 1: Basic JAX Implementation (lax.scan)

### Step 1.1: GRU Cell with lax.scan

```python
def gru_cell(carry, x_t, Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh):
    """Full non-associative GRU step."""
    h = carry

    # Gates (depend on h!)
    z = jax.nn.sigmoid(x_t @ Wz + h @ Uz + bz)  # update gate
    r = jax.nn.sigmoid(x_t @ Wr + h @ Ur + br)  # reset gate

    # Candidate (depends on r * h)
    h_candidate = jnp.tanh(x_t @ Wh + (r * h) @ Uh + bh)

    # Update
    h_new = (1 - z) * h + z * h_candidate
    return h_new, h_new

def gru_sequence(x, params):
    """GRU over sequence using lax.scan."""
    B, S, D = x.shape
    h0 = jnp.zeros((B, params['Uz'].shape[0]))

    def step(h, x_t):
        return gru_cell(h, x_t, **params)

    # Scan over time dimension
    x_seq = x.transpose(1, 0, 2)  # (S, B, D)
    _, h_seq = jax.lax.scan(step, h0, x_seq)
    return h_seq.transpose(1, 0, 2)  # (B, S, hidden)
```

### Step 1.2: Parallel EMA

```python
def parallel_ema(x, alpha):
    """
    O(S) parallel EMA computation.

    ema_t = α * x_t + (1-α) * ema_{t-1}

    Closed form via cumsum:
    ema_t = α * Σ_{i=0}^{t} (1-α)^{t-i} * x_i
    """
    B, S, D = x.shape
    decay = 1.0 - alpha

    # Time indices
    t = jnp.arange(S, dtype=x.dtype)

    # Weight each input by inverse decay power, cumsum, then apply forward decay
    decay_powers = decay ** t  # (S,)
    inv_decay_powers = decay ** (-t)  # (S,)

    # x_weighted[t] = x[t] / decay^t
    x_weighted = x * inv_decay_powers[None, :, None]

    # Cumulative sum
    x_cumsum = jnp.cumsum(x_weighted, axis=1)

    # Apply forward decay and scale
    ema = alpha * x_cumsum * decay_powers[None, :, None]

    return ema
```

### Step 1.3: Combined Layer

```python
class GRUEMACell(nn.Module):
    hidden_dim: int
    ema_alpha: float = 0.01
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        B, S, D = x.shape

        # GRU parameters
        Wz = self.param('Wz', nn.initializers.glorot_uniform(), (D, self.hidden_dim))
        Wr = self.param('Wr', nn.initializers.glorot_uniform(), (D, self.hidden_dim))
        Wh = self.param('Wh', nn.initializers.glorot_uniform(), (D, self.hidden_dim))
        Uz = self.param('Uz', nn.initializers.orthogonal(), (self.hidden_dim, self.hidden_dim))
        Ur = self.param('Ur', nn.initializers.orthogonal(), (self.hidden_dim, self.hidden_dim))
        Uh = self.param('Uh', nn.initializers.orthogonal(), (self.hidden_dim, self.hidden_dim))
        bz = self.param('bz', nn.initializers.zeros, (self.hidden_dim,))
        br = self.param('br', nn.initializers.zeros, (self.hidden_dim,))
        bh = self.param('bh', nn.initializers.zeros, (self.hidden_dim,))

        # GRU forward
        h_gru = gru_sequence(x, {
            'Wz': Wz, 'Wr': Wr, 'Wh': Wh,
            'Uz': Uz, 'Ur': Ur, 'Uh': Uh,
            'bz': bz, 'br': br, 'bh': bh
        })

        # EMA forward (on projected input)
        x_ema = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        ema = parallel_ema(x_ema, self.ema_alpha)

        # Learnable combination
        ema_weight = self.param('ema_weight', nn.initializers.ones, (self.hidden_dim,))

        return h_gru + ema_weight * ema
```

---

## Phase 2: Pallas Kernel (FlashRNN-like)

### Why Pallas?

`lax.scan` compiles well but can't match hand-tuned CUDA for sequential RNNs because:
- Can't use persistent kernels (state in registers across steps)
- Memory bandwidth limited (reload weights each step)

Pallas lets us write Triton-style kernels in JAX.

### Step 2.1: Fused GRU Kernel Design

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

@pl.kernel
def fused_gru_fwd_kernel(
    # Inputs
    x_ref,      # (B, S, D) input
    Wz_ref,     # (D, H) weight
    Wr_ref, Wh_ref, Uz_ref, Ur_ref, Uh_ref,
    bz_ref, br_ref, bh_ref,
    # Outputs
    h_out_ref,  # (B, S, H) output
    # Block sizes
    BLOCK_B: pl.constexpr,
    BLOCK_S: pl.constexpr,  # Chunk size for tiling
    BLOCK_H: pl.constexpr,
):
    """
    Fused GRU forward pass.

    Key optimizations:
    1. Keep h in registers across time steps
    2. Tile over batch and hidden dimensions
    3. Prefetch weights to shared memory
    """
    batch_idx = pl.program_id(0)
    hidden_block = pl.program_id(1)

    # Load weights into shared memory (once per block)
    # ... weight loading ...

    # Initialize hidden state in registers
    h = jnp.zeros((BLOCK_B, BLOCK_H))

    # Sequential loop over time (within kernel)
    for t in range(x_ref.shape[1]):
        # Load input slice
        x_t = pl.load(x_ref, (batch_idx, t, pl.dslice(None)))

        # Compute gates (fused matmuls + activations)
        z = sigmoid(x_t @ Wz + h @ Uz + bz)
        r = sigmoid(x_t @ Wr + h @ Ur + br)
        h_cand = tanh(x_t @ Wh + (r * h) @ Uh + bh)

        # Update hidden (stays in registers!)
        h = (1 - z) * h + z * h_cand

        # Store output
        pl.store(h_out_ref, (batch_idx, t, hidden_block), h)
```

### Step 2.2: Backward Kernel (Gradient Checkpointing)

```python
@pl.kernel
def fused_gru_bwd_kernel(
    # Forward inputs (for recomputation)
    x_ref, weights_refs...,
    # Gradients
    dh_out_ref,  # (B, S, H) upstream gradient
    # Outputs
    dx_ref, dW_refs...,
):
    """
    Backward pass with activation recomputation.

    FlashRNN-style: don't store intermediate activations,
    recompute h[t] during backward pass.
    """
    # Recompute forward pass to get h[t] at each step
    # Then compute gradients in reverse
    ...
```

### Step 2.3: Wrapper Function

```python
def fused_gru_ema(x, params, ema_alpha=0.01):
    """
    Fused GRU + EMA using Pallas kernels.
    """
    # GRU via Pallas kernel
    h_gru = fused_gru_fwd_kernel(x, **params)

    # EMA is already O(S) parallel, use JAX cumsum
    ema = parallel_ema(x @ params['W_ema'], ema_alpha)

    # Combine
    return h_gru + params['ema_weight'] * ema
```

---

## Phase 3: gruboros Data Pipeline in JAX

### Step 3.1: Memory-Mapped Document Iterator

```python
import mmap
import numpy as np

class MMapDocumentIterator:
    """
    Memory-mapped file iterator for gruboros-style document streaming.

    Key features:
    - mmap'd access (no full file load)
    - Document boundary tracking
    - Chunk-based iteration for long documents
    """

    def __init__(self, path, chunk_size=2048, tokenizer=None):
        self.path = path
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        # Memory map the file
        with open(path, 'rb') as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self.file_size = len(self.mm)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= self.file_size:
            raise StopIteration

        # Read chunk
        end = min(self.position + self.chunk_size * 4, self.file_size)  # ~4 bytes per token estimate
        text = self.mm[self.position:end].decode('utf-8', errors='ignore')

        # Tokenize
        tokens = self.tokenizer.encode(text)[:self.chunk_size]

        # Update position
        self.position = end

        return np.array(tokens, dtype=np.int32)
```

### Step 3.2: JAX-Compatible DataLoader

```python
import grain.python as grain

class GruborosDataSource(grain.RandomAccessDataSource):
    """Grain-compatible data source for mmap'd documents."""

    def __init__(self, path, chunk_size, tokenizer):
        self.iterator = MMapDocumentIterator(path, chunk_size, tokenizer)
        # Pre-compute chunks for random access
        self._index_file()

    def _index_file(self):
        """Build index of chunk positions."""
        self.chunk_positions = []
        # ... index building ...

    def __len__(self):
        return len(self.chunk_positions)

    def __getitem__(self, idx):
        pos = self.chunk_positions[idx]
        # Seek and read chunk
        ...

def create_dataloader(path, batch_size, chunk_size, tokenizer):
    """Create Grain dataloader for JAX."""
    source = GruborosDataSource(path, chunk_size, tokenizer)

    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=4,
        worker_buffer_size=2,
    )

    return loader
```

---

## Phase 4: Distributed Training (FSDP)

### Step 4.1: Mesh Setup

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

def create_mesh(num_devices=8):
    """Create device mesh for FSDP."""
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=('fsdp',))
    return mesh

# Partition specs
param_spec = P('fsdp')  # Shard params across devices
data_spec = P('fsdp')   # Shard batch across devices
```

### Step 4.2: Training Step

```python
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh,
         in_specs=(param_spec, data_spec, P()),
         out_specs=(param_spec, P()))
def train_step(params, batch, rng):
    """FSDP-sharded training step."""

    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'])
        loss = cross_entropy(logits, batch['target_ids'], batch['mask'])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Sync gradients across FSDP shards
    grads = jax.lax.pmean(grads, axis_name='fsdp')

    # Update params
    params = optax_update(params, grads)

    return params, loss
```

---

## Phase 5: Full Training Loop

```python
def train(config):
    # Setup
    mesh = create_mesh(config.num_devices)
    tokenizer = tiktoken.get_encoding('p50k_base')
    dataloader = create_dataloader(config.data_path, config.batch_size,
                                   config.chunk_size, tokenizer)

    # Model
    model = GRUEMALMModel(config.model)

    # Initialize params (sharded)
    with mesh:
        rng = jax.random.PRNGKey(config.seed)
        params = model.init(rng, jnp.ones((1, config.chunk_size), dtype=jnp.int32))
        params = shard_params(params, axis_name='fsdp')

    # Optimizer
    optimizer = optax.adamw(config.lr, weight_decay=config.weight_decay)
    opt_state = optimizer.init(params)

    # Training loop
    for step, batch in enumerate(dataloader):
        batch = jax.device_put(batch, data_sharding)

        with mesh:
            params, loss = train_step(params, batch, rng)

        if step % config.log_every == 0:
            print(f"Step {step}: loss = {loss:.4f}")

        if step % config.save_every == 0:
            save_checkpoint(params, step)
```

---

## File Structure

```
gruboros/
├── jax_gru_ema/
│   ├── __init__.py
│   ├── model/
│   │   ├── gru_ema_cell.py      # GRU + EMA cell (lax.scan version)
│   │   ├── gru_ema_layer.py     # Full layer with up/down proj
│   │   ├── gru_ema_block.py     # Block with LayerNorm + residual
│   │   ├── gru_ema_lm.py        # Full LM model
│   │   └── pallas/
│   │       ├── fused_gru_fwd.py # Pallas forward kernel
│   │       └── fused_gru_bwd.py # Pallas backward kernel
│   ├── data/
│   │   ├── mmap_iterator.py     # Memory-mapped document iterator
│   │   └── grain_loader.py      # Grain dataloader wrapper
│   ├── distributed/
│   │   ├── mesh.py              # Device mesh setup
│   │   ├── sharding.py          # FSDP sharding utils
│   │   └── sync.py              # Gradient sync
│   └── train.py                 # Main training script
├── train_jax.sh                 # Launch script
└── JAX_GRU_EMA_WIP.md          # This file
```

---

## TODO

- [x] Phase 1: Basic lax.scan GRU + EMA (COMPLETE 2024-12-03)
  - [x] Implement GRU cell with lax.scan (non-associative, full reset gate)
  - [x] Implement parallel EMA via cumsum trick with log-space stability
  - [x] GRUEMALayer with up/down projections and gating
  - [x] GRUEMABlock with pre-LayerNorm and residual
  - [x] Full GRUEMALMModel with tied embeddings
  - [x] All tests pass on GPU (8x CUDA devices)
  - [x] bf16 support (orthogonal init workaround)
  - [x] Verified EMA correctness: max error < 1e-6 vs sequential
  - [x] Verified gradient flow: no vanishing gradients through 256 steps
  - [x] Scaling test: tiny=15.2M, small=44.7M, medium=164.8M params

- [ ] Phase 2: Integration with xlstm-jax trainer
  - [ ] Port trainer config
  - [ ] Port distributed utilities
  - [ ] Test FSDP on 8 GPUs

- [ ] Phase 3: gruboros data pipeline
  - [ ] Memory-mapped iterator
  - [ ] Grain dataloader
  - [ ] Document boundary handling

- [ ] Phase 4: Pallas kernels (performance)
  - [ ] Forward kernel design
  - [ ] Backward kernel with recomputation
  - [ ] Benchmark vs lax.scan
  - [ ] Benchmark vs FlashRNN (PyTorch)

- [ ] Phase 5: Scale up
  - [ ] 1B parameter model
  - [ ] Multi-node training
  - [ ] Long context (8K+ tokens)

---

## References

- [xlstm-jax](https://github.com/NX-AI/xlstm-jax) - NXAI's JAX implementation
- [Pallas documentation](https://jax.readthedocs.io/en/latest/pallas/) - JAX GPU kernel authoring
- [FlashRNN](https://github.com/NX-AI/flashrnn) - Original fused RNN kernels
- [Grain](https://github.com/google/grain) - JAX-native data loading
- [MinGRU paper](https://arxiv.org/abs/2410.01201) - Parallel GRU via associative scan (different approach)

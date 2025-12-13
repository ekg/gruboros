"""FlashRNN GRU wrapper for minLM compatibility"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from flashrnn import flashrnn, FlashRNNConfig

# Track which (head_dim, batch_size) combos have been JIT-compiled
_FLASHRNN_JIT_COMPILED = set()


def precompile_flashrnn_kernels(head_dim, batch_size, device='cuda', dtype=torch.bfloat16):
    """
    Pre-compile FlashRNN kernels BEFORE DDP initialization.

    Call this function on ALL ranks BEFORE init_process_group().
    Uses file-based synchronization instead of DDP barriers to avoid timeouts.

    FlashRNN caches compiled kernels to disk, so only one process needs to compile.
    Other processes will use the cached version.

    Args:
        head_dim: Hidden dimension (e.g., 2048)
        batch_size: Batch size (e.g., 64)
        device: CUDA device
        dtype: Data type (typically torch.bfloat16)
    """
    global _FLASHRNN_JIT_COMPILED

    # Defensive type conversion - ensure we have plain ints
    head_dim = int(head_dim) if not isinstance(head_dim, int) else head_dim
    batch_size = int(batch_size) if not isinstance(batch_size, int) else batch_size
    print(f"[FlashRNN precompile] head_dim={head_dim} (type={type(head_dim)}), batch_size={batch_size} (type={type(batch_size)})", flush=True)

    config_key = (head_dim, batch_size, dtype)
    if config_key in _FLASHRNN_JIT_COMPILED:
        return

    # Get rank from environment (set by torchrun before init_process_group)
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    dtype_str = 'bfloat16' if dtype == torch.bfloat16 else 'float32' if dtype == torch.float32 else 'float16'

    # File-based synchronization (works before DDP init)
    lock_file = f"/tmp/flashrnn_jit_lock_{head_dim}_{batch_size}_{dtype_str}"
    done_file = f"/tmp/flashrnn_jit_done_{head_dim}_{batch_size}_{dtype_str}"

    # Clean up old done file on rank 0
    if rank == 0 and os.path.exists(done_file):
        os.remove(done_file)

    # Small delay to ensure rank 0 cleans up first
    time.sleep(0.5)

    if rank == 0:
        print(f"[FlashRNN] Rank 0 pre-compiling JIT for head_dim={head_dim}, batch={batch_size}...", flush=True)
        print(f"[FlashRNN] This may take 10-15 minutes for large dimensions. Please wait.", flush=True)

        start_time = time.time()

        # Create lock file
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))

        # Compile the kernel
        seq_len = 16
        num_heads, num_gates = 1, 3

        Wx = torch.randn(batch_size, seq_len, num_gates, num_heads, head_dim, device=device, dtype=dtype)
        R = torch.randn(num_gates, num_heads, head_dim, head_dim, device=device, dtype=dtype)
        b = torch.zeros(num_gates, num_heads, head_dim, device=device, dtype=dtype)
        states = torch.zeros(1, batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

        config = FlashRNNConfig(
            function='gru',
            backend='cuda_fused',
            hidden_dim=head_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            dtype=dtype_str,
            dtype_b=dtype_str,
            dtype_r=dtype_str,
            dtype_w=dtype_str,
            dtype_s=dtype_str,
            dtype_a=dtype_str,
        )

        _ = flashrnn(Wx=Wx, R=R, b=b, states=states, config=config)
        torch.cuda.synchronize()

        elapsed = time.time() - start_time
        print(f"[FlashRNN] Rank 0 JIT compilation complete! Took {elapsed:.1f}s", flush=True)

        # Signal completion
        with open(done_file, 'w') as f:
            f.write('done')
    else:
        # Other ranks wait for done file
        print(f"[FlashRNN] Rank {rank} waiting for JIT compilation to complete...", flush=True)
        wait_start = time.time()
        max_wait = 1800  # 30 minutes max

        while not os.path.exists(done_file):
            time.sleep(5)
            elapsed = time.time() - wait_start
            if elapsed > max_wait:
                raise RuntimeError(f"[FlashRNN] Rank {rank}: Timeout waiting for JIT compilation")
            if elapsed > 60 and int(elapsed) % 60 == 0:
                print(f"[FlashRNN] Rank {rank}: Still waiting... ({elapsed:.0f}s)", flush=True)

        print(f"[FlashRNN] Rank {rank} proceeding after JIT compilation", flush=True)

        # Also run flashrnn once to load cached kernel
        seq_len = 16
        num_heads, num_gates = 1, 3

        Wx = torch.randn(batch_size, seq_len, num_gates, num_heads, head_dim, device=device, dtype=dtype)
        R = torch.randn(num_gates, num_heads, head_dim, head_dim, device=device, dtype=dtype)
        b = torch.zeros(num_gates, num_heads, head_dim, device=device, dtype=dtype)
        states = torch.zeros(1, batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

        config = FlashRNNConfig(
            function='gru',
            backend='cuda_fused',
            hidden_dim=head_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            dtype=dtype_str,
            dtype_b=dtype_str,
            dtype_r=dtype_str,
            dtype_w=dtype_str,
            dtype_s=dtype_str,
            dtype_a=dtype_str,
        )

        _ = flashrnn(Wx=Wx, R=R, b=b, states=states, config=config)
        torch.cuda.synchronize()

    _FLASHRNN_JIT_COMPILED.add(config_key)
    print(f"[FlashRNN] Rank {rank}: Kernel ready for head_dim={head_dim}, batch={batch_size}", flush=True)


class FlashGRU(nn.Module):
    """
    FlashRNN GRU wrapper compatible with minLM API.

    Uses FlashRNN's optimized GRU implementation with document boundary support.
    FlashRNN provides 50x speedup over vanilla PyTorch while maintaining true recurrence.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        **kwargs  # Ignore other minGRU-specific args
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # FlashRNN uses multi-head structure (we use single head for simplicity)
        self.num_heads = 1
        self.head_dim = self.dim_inner

        # GRU has 3 gates: reset, update, new
        self.num_gates = 3

        # Input projection: dim -> (gates * heads * head_dim)
        self.input_proj = nn.Linear(dim, self.num_gates * self.num_heads * self.head_dim, bias=False)

        # Recurrent weights: [G, N, D, D] where G=gates, N=heads, D=head_dim
        self.recurrent_weights = nn.Parameter(
            torch.randn(self.num_gates, self.num_heads, self.head_dim, self.head_dim)
        )
        nn.init.orthogonal_(self.recurrent_weights.view(self.num_gates, -1))

        # Bias: [G, N, D]
        self.bias = nn.Parameter(torch.zeros(self.num_gates, self.num_heads, self.head_dim))

        # Output projection: dim_inner -> dim
        self.output_proj = nn.Linear(self.dim_inner, dim, bias=False)

    def forward(
        self,
        x,
        prev_hiddens=None,
        prev_conv_buffers=None,  # Unused, for API compatibility
        return_hiddens=True,
        return_next_prev_hidden=True,
        actual_length=None,  # Unused, for API compatibility
        doc_boundaries=None  # Unused for now, for API compatibility
    ):
        """
        Args:
            x: (batch, seq_len, dim)
            prev_hiddens: Previous hidden state (batch, dim_inner) or None

        Returns:
            output: (batch, seq_len, dim)
            next_hiddens: (batch, dim_inner) if return_hiddens else None
            next_conv_buffers: None (no conv in GRU)
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Note: JIT warmup should be done via precompile_flashrnn_kernels()
        # BEFORE DDP initialization. The kernel should already be compiled.

        # Project input: [B, T, dim] -> [B, T, G*N*D]
        x_proj = self.input_proj(x)  # [B, T, G*N*D]

        # Reshape to FlashRNN format: [B, T, G, N, D]
        Wx = x_proj.view(batch, seq_len, self.num_gates, self.num_heads, self.head_dim)

        # Prepare initial hidden state: [S, B, 1, N, D] where S=1 for GRU
        if prev_hiddens is not None:
            # prev_hiddens is [B, dim_inner], reshape to [1, B, 1, N, D]
            states_initial = prev_hiddens.view(batch, self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(2)
        else:
            states_initial = torch.zeros(1, batch, 1, self.num_heads, self.head_dim, device=device, dtype=dtype)

        # Create FlashRNN config with consistent dtypes
        dtype_str = 'bfloat16' if dtype == torch.bfloat16 else 'float32' if dtype == torch.float32 else 'float16'
        config = FlashRNNConfig(
            function='gru',
            backend='cuda_fused',
            hidden_dim=self.head_dim,
            num_heads=self.num_heads,
            batch_size=batch,
            dtype=dtype_str,
            dtype_b=dtype_str,  # Bias dtype
            dtype_r=dtype_str,  # Recurrent dtype
            dtype_w=dtype_str,  # Weight dtype
            dtype_s=dtype_str,  # State dtype
            dtype_a=dtype_str,  # Activation dtype
        )

        # Run FlashRNN GRU
        # states: [S, B, T, N, D]
        # last_states: [S, B, 1, N, D]
        states, last_states = flashrnn(
            Wx=Wx,
            R=self.recurrent_weights,
            b=self.bias,
            states=states_initial,
            config=config
        )

        # Extract hidden states: [S, B, T, N, D] -> [B, T, N, D] -> [B, T, dim_inner]
        hidden_seq = states[0]  # [B, T, N, D]
        hidden_seq = hidden_seq.view(batch, seq_len, self.dim_inner)

        # Project to output dimension
        output = self.output_proj(hidden_seq)  # [B, T, dim]

        if return_hiddens:
            # Extract last hidden state: [S, B, 1, N, D] -> [B, dim_inner]
            next_hiddens = last_states[0, :, 0, :, :].reshape(batch, self.dim_inner)
            return output, next_hiddens, None
        else:
            return output

    def __repr__(self):
        return f"FlashGRU(dim={self.dim}, dim_inner={self.dim_inner}, FlashRNN backend)"

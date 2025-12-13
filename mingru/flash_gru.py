"""FlashRNN GRU wrapper for minLM compatibility"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from flashrnn import flashrnn, FlashRNNConfig

# NOTE: We use standard 'gru' function with FlashRNN's built-in 'cuda' backend.
# The 'cuda' (alternating) backend works well with multi-head GRU configuration.
# Key insight: Use many small heads (32 heads × 64 head_dim = 2048) to fit kernel constraints.

# Track which (head_dim, batch_size) combos have been JIT-compiled
_FLASHRNN_JIT_COMPILED = set()


def precompile_flashrnn_kernels(dim_inner, batch_size, device='cuda', dtype=torch.bfloat16, num_heads=None):
    """
    Pre-compile FlashRNN kernels BEFORE DDP initialization.

    Call this function on ALL ranks BEFORE init_process_group().
    Uses file-based synchronization instead of DDP barriers to avoid timeouts.

    FlashRNN caches compiled kernels to disk, so only one process needs to compile.
    Other processes will use the cached version.

    Args:
        dim_inner: Total hidden dimension (e.g., 2048) - will be split across heads
        batch_size: Batch size (e.g., 64)
        device: CUDA device
        dtype: Data type (typically torch.bfloat16)
        num_heads: Number of heads (auto-calculated if None)
    """
    global _FLASHRNN_JIT_COMPILED

    # Defensive type conversion - ensure we have plain ints
    dim_inner = int(dim_inner) if not isinstance(dim_inner, int) else dim_inner
    batch_size = int(batch_size) if not isinstance(batch_size, int) else batch_size

    # Auto-calculate num_heads to keep head_dim within kernel limits
    # Based on official FlashRNN benchmarks: working configs use (DH,NH) like (64,12), (32,24)
    # Larger head_dim (512+) exceeds shared memory limits on A100
    MAX_HEAD_DIM = 64
    if num_heads is None:
        if dim_inner <= MAX_HEAD_DIM:
            num_heads = 1
        else:
            num_heads = (dim_inner + MAX_HEAD_DIM - 1) // MAX_HEAD_DIM
            while dim_inner % num_heads != 0:
                num_heads += 1

    head_dim = dim_inner // num_heads

    print(f"[FlashRNN precompile] dim_inner={dim_inner}, num_heads={num_heads}, head_dim={head_dim}, batch_size={batch_size}", flush=True)

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

        # Compile the kernel with 3 gates (standard GRU)
        # Use the num_heads calculated above - don't override to 1!
        seq_len = 16
        num_gates = 3  # Standard GRU has 3 gates: r, z, n

        Wx = torch.randn(batch_size, seq_len, num_gates, num_heads, head_dim, device=device, dtype=dtype)
        R = torch.randn(num_gates, num_heads, head_dim, head_dim, device=device, dtype=dtype)
        b = torch.zeros(num_gates, num_heads, head_dim, device=device, dtype=dtype)
        states = torch.zeros(1, batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

        config = FlashRNNConfig(
            function='gru',  # Standard GRU - FlashRNN handles backend selection
            backend='cuda_fused',  # Will auto-downgrade to 'cuda' for GRU
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

        # Also run flashrnn once to load cached kernel with 3 gates
        # Use the num_heads calculated above - don't override to 1!
        seq_len = 16
        num_gates = 3  # Standard GRU has 3 gates: r, z, n

        Wx = torch.randn(batch_size, seq_len, num_gates, num_heads, head_dim, device=device, dtype=dtype)
        R = torch.randn(num_gates, num_heads, head_dim, head_dim, device=device, dtype=dtype)
        b = torch.zeros(num_gates, num_heads, head_dim, device=device, dtype=dtype)
        states = torch.zeros(1, batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

        config = FlashRNNConfig(
            function='gru',  # Standard GRU - FlashRNN handles backend selection
            backend='cuda_fused',  # Will auto-downgrade to 'cuda' for GRU
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

    IMPORTANT FIX: FlashRNN fused kernel expects 4 gates, not 3!
    Gate layout for fused kernel:
    - Gate 0 (graw): recurrent part for candidate (Un*h_prev) - Wx[0]=0, R[0]=Un
    - Gate 1 (rraw): reset gate - Wx[1]=Wr*x, R[1]=Ur
    - Gate 2 (zraw): update gate - Wx[2]=Wz*x, R[2]=Uz
    - Gate 3 (nraw): input part for candidate - Wx[3]=Wn*x, R[3]=0

    This padding enables cuda_fused backend (tensor core optimized) instead of
    auto-downgrading to cuda/alternating backend which deadlocks on large dims.

    MULTI-HEAD SUPPORT: For large dims (>1024), use multiple heads with smaller
    head_dim to stay within FlashRNN kernel constraints. The kernel requires
    head_dim to fit in shared memory/registers.
    """

    def __init__(
        self,
        dim,
        expansion_factor=1.0,
        num_heads=None,  # Auto-calculate if None
        **kwargs  # Ignore other minGRU-specific args
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Auto-calculate num_heads to keep head_dim within kernel limits
        # Based on official FlashRNN benchmarks: working configs use (DH,NH) like (64,12), (32,24)
        # Larger head_dim (512+) exceeds shared memory limits on A100
        MAX_HEAD_DIM = 64
        if num_heads is None:
            if self.dim_inner <= MAX_HEAD_DIM:
                num_heads = 1
            else:
                # Find smallest num_heads that keeps head_dim <= MAX_HEAD_DIM
                num_heads = (self.dim_inner + MAX_HEAD_DIM - 1) // MAX_HEAD_DIM
                # Make sure dim_inner is divisible by num_heads
                while self.dim_inner % num_heads != 0:
                    num_heads += 1

        self.num_heads = num_heads
        self.head_dim = self.dim_inner // num_heads

        assert self.dim_inner % self.num_heads == 0, \
            f"dim_inner ({self.dim_inner}) must be divisible by num_heads ({self.num_heads})"

        print(f"[FlashGRU] dim={dim}, dim_inner={self.dim_inner}, num_heads={self.num_heads}, head_dim={self.head_dim}", flush=True)

        # GRU uses 3 gates: r (reset), z (update), n (candidate)
        # Note: cuda_fused backend doesn't work with GRU due to 3-gate constraint
        # FlashRNN auto-downgrades to "cuda" backend for function="gru"
        self.num_gates = 3

        # Input projection: dim -> (3 * heads * head_dim) for r, z, n gates
        self.input_proj = nn.Linear(dim, self.num_gates * self.num_heads * self.head_dim, bias=False)

        # Recurrent weights: [3, N, D, D] for all 3 GRU gates
        self.recurrent_weights = nn.Parameter(
            torch.randn(self.num_gates, self.num_heads, self.head_dim, self.head_dim)
        )
        nn.init.orthogonal_(self.recurrent_weights.view(self.num_gates, -1))

        # Bias: [4, N, D] - 4 gates for kernel, but gate 0 is unused (set to 0)
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

        # Project input: [B, T, dim] -> [B, T, 3*N*D] for r, z, n gates
        x_proj = self.input_proj(x)  # [B, T, 3*N*D]

        # Reshape to [B, T, 3, N, D] - FlashRNN expects gates dimension before heads
        Wx = x_proj.view(batch, seq_len, self.num_gates, self.num_heads, self.head_dim)

        # Prepare initial hidden state: [1, B, 1, N, D] where first dim is num_states=1 for GRU
        if prev_hiddens is not None:
            # prev_hiddens is [B, dim_inner], reshape to [1, B, 1, N, D]
            states_initial = prev_hiddens.view(batch, self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(2)
        else:
            states_initial = torch.zeros(1, batch, 1, self.num_heads, self.head_dim, device=device, dtype=dtype)

        # Create FlashRNN config
        # Use function="gru" which auto-selects the appropriate backend
        # cuda_fused doesn't work with GRU's 3 gates, FlashRNN will use "cuda" backend
        dtype_str = 'bfloat16' if dtype == torch.bfloat16 else 'float32' if dtype == torch.float32 else 'float16'
        config = FlashRNNConfig(
            function='gru',  # Standard GRU function - FlashRNN handles backend selection
            backend='cuda_fused',  # Will auto-downgrade to 'cuda' for GRU
            hidden_dim=self.head_dim,
            num_heads=self.num_heads,
            batch_size=batch,
            dtype=dtype_str,
            dtype_b=dtype_str,
            dtype_r=dtype_str,
            dtype_w=dtype_str,
            dtype_s=dtype_str,
            dtype_a=dtype_str,
        )

        # Run FlashRNN GRU with 3-gate tensors
        # Output states: [1, B, T, N, D]
        # last_states: [1, B, 1, N, D]
        states, last_states = flashrnn(
            Wx=Wx,
            R=self.recurrent_weights,  # [3, N, D, D]
            b=self.bias,  # [3, N, D]
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

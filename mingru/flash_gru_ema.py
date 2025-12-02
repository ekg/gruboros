"""
FlashGRU + EMA: Fast GRU via FlashRNN + Long-range memory via parallel EMA.

Architecture:
  - h_fast = FlashRNN_GRU(x)       # 50x faster than vanilla GRU
  - h_slow = parallel_EMA(x)      # O(1) depth for long-range context
  - output = W_fast @ h_fast + W_slow @ h_slow

This gives us:
- Fast local pattern recognition (FlashRNN GRU, hardware-optimized)
- Long-range memory (EMA with ~70 token half-life at alpha=0.01)
"""

import torch
import torch.nn as nn
import math

try:
    from flashrnn import flashrnn, FlashRNNConfig
    FLASHRNN_AVAILABLE = True
except ImportError:
    print("Warning: FlashRNN not available. Install with: pip install flashrnn")
    FLASHRNN_AVAILABLE = False


def parallel_ema_chunk(x, decay, alpha, h0=None):
    """Compute EMA for a chunk using parallel cumsum trick."""
    B, T, D = x.shape
    device = x.device

    if h0 is None:
        h0 = torch.zeros(B, D, device=device, dtype=torch.float32)

    x_fp32 = x.float()
    h0_fp32 = h0.float()
    decay = float(decay)
    alpha = float(alpha)

    t_idx = torch.arange(T, device=device, dtype=torch.float32)
    h0_decay_powers = decay ** (t_idx + 1)
    inv_decay_powers = decay ** (-t_idx)

    scaled_x = x_fp32 * inv_decay_powers.view(1, T, 1)
    cumsum = scaled_x.cumsum(dim=1)
    input_contribution = alpha * cumsum * (decay ** t_idx).view(1, T, 1)
    h0_contribution = h0_fp32.unsqueeze(1) * h0_decay_powers.view(1, T, 1)

    h_all = h0_contribution + input_contribution
    return h_all, h_all[:, -1]


def parallel_ema(x, decay, alpha, h0=None, chunk_size=64):
    """Compute EMA with chunking for numerical stability."""
    B, T, D = x.shape
    device = x.device
    dtype = x.dtype

    if h0 is None:
        h0 = torch.zeros(B, D, device=device, dtype=dtype)

    if T <= chunk_size:
        h_all, h_final = parallel_ema_chunk(x, decay, alpha, h0)
        return h_all.to(dtype), h_final.to(dtype)

    all_outputs = []
    h_current = h0.float()

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        x_chunk = x[:, start:end]
        h_chunk, h_current = parallel_ema_chunk(x_chunk, decay, alpha, h_current)
        all_outputs.append(h_chunk)

    h_all = torch.cat(all_outputs, dim=1)
    return h_all.to(dtype), h_current.to(dtype)


class FlashGRU_EMA(nn.Module):
    """
    FlashRNN GRU + Parallel EMA for fast recurrence with long-range memory.

    - GRU path: FlashRNN's optimized cuda_fused backend (50x speedup)
    - EMA path: Parallel cumsum for O(1) depth long-range memory

    Combined output provides both local pattern recognition and long-range context.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ema_dim: int = None,
        ema_alpha: float = 0.01,
        learnable_alpha: bool = True,
        z_bias_input: float = 0.0,  # Unused, for API compat
        z_bias_hidden: float = 0.0,  # Unused, for API compat
        recurrence_chunk_size: int = 64,  # For EMA chunking
        **kwargs
    ):
        super().__init__()

        if not FLASHRNN_AVAILABLE:
            raise ImportError("FlashRNN not available. Install with: pip install flashrnn")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ema_dim = ema_dim if ema_dim is not None else dim
        self.ema_chunk_size = recurrence_chunk_size

        # === FlashRNN GRU (fast path) ===
        self.num_heads = 1
        self.head_dim = self.dim_inner
        self.num_gates = 3  # GRU: reset, update, new

        # Input projection for GRU
        self.input_proj = nn.Linear(dim, self.num_gates * self.num_heads * self.head_dim, bias=False)

        # Recurrent weights: [G, N, D, D]
        self.recurrent_weights = nn.Parameter(
            torch.randn(self.num_gates, self.num_heads, self.head_dim, self.head_dim)
        )
        nn.init.orthogonal_(self.recurrent_weights.view(self.num_gates, -1))

        # Bias: [G, N, D]
        self.bias = nn.Parameter(torch.zeros(self.num_gates, self.num_heads, self.head_dim))

        # Output projection for GRU
        if expansion_factor != 1.0:
            self.to_out_fast = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out_fast = nn.Identity()

        # === EMA (slow path) ===
        self.ema_in_proj = nn.Linear(dim, self.ema_dim, bias=False)
        self.ema_out_proj = nn.Linear(self.ema_dim, dim, bias=False)

        # EMA alpha (learnable in logit space)
        if learnable_alpha:
            init_logit = math.log(ema_alpha / (1 - ema_alpha))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('alpha_logit', torch.tensor(math.log(ema_alpha / (1 - ema_alpha))))

        self._init_weights()

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _init_weights(self):
        # GRU output - zero init for residual
        if not isinstance(self.to_out_fast, nn.Identity):
            nn.init.constant_(self.to_out_fast.weight, 0.0)

        # EMA projections - small init
        nn.init.normal_(self.ema_in_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.ema_out_proj.weight, 0.0)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        # Use model weight dtype (not input dtype) for FlashRNN config
        # This ensures correct kernel compilation under autocast
        dtype = self.input_proj.weight.dtype
        if x.dtype != dtype:
            x = x.to(dtype)

        # Parse hidden states: [h_fast | h_slow]
        if prev_hidden is None:
            h_fast = None
            h_slow = torch.zeros(B, self.ema_dim, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner].contiguous()
            h_slow = prev_hidden[:, self.dim_inner:].contiguous()

        alpha = self.alpha
        decay = 1.0 - alpha

        # === FlashRNN GRU (fast path) ===
        # Project input: [B, T, dim] -> [B, T, G*N*D]
        x_proj = self.input_proj(x)
        Wx = x_proj.view(B, T, self.num_gates, self.num_heads, self.head_dim)

        # Initial hidden state: [S=1, B, 1, N, D]
        if h_fast is not None:
            states_initial = h_fast.view(B, self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(2)
        else:
            states_initial = torch.zeros(1, B, 1, self.num_heads, self.head_dim, device=device, dtype=dtype)

        # FlashRNN config
        dtype_str = 'bfloat16' if dtype == torch.bfloat16 else 'float32' if dtype == torch.float32 else 'float16'
        config = FlashRNNConfig(
            function='gru',
            backend='cuda_fused',
            hidden_dim=self.head_dim,
            num_heads=self.num_heads,
            batch_size=B,
            dtype=dtype_str,
            dtype_b=dtype_str,
            dtype_r=dtype_str,
            dtype_w=dtype_str,
            dtype_s=dtype_str,
            dtype_a=dtype_str,
        )

        # Run FlashRNN
        states, last_states = flashrnn(
            Wx=Wx,
            R=self.recurrent_weights,
            b=self.bias,
            states=states_initial,
            config=config
        )

        # Extract hidden sequence: [1, B, T, N, D] -> [B, T, dim_inner]
        h_fast_seq = states[0].view(B, T, self.dim_inner)
        h_fast_final = last_states[0, :, 0, :, :].reshape(B, self.dim_inner)

        # === Parallel EMA (slow path) ===
        x_ema = self.ema_in_proj(x)

        if doc_boundaries is None:
            h_slow_seq, h_slow_final = parallel_ema(
                x_ema, decay, alpha, h_slow, chunk_size=self.ema_chunk_size
            )
        else:
            # With doc boundaries, fall back to sequential
            h_slow_list = []
            h_slow_t = h_slow
            for t in range(T):
                if doc_boundaries[:, t].any():
                    h_slow_t = h_slow_t.masked_fill(doc_boundaries[:, t].unsqueeze(-1), 0.0)
                h_slow_t = decay * h_slow_t + alpha * x_ema[:, t]
                h_slow_list.append(h_slow_t)
            h_slow_seq = torch.stack(h_slow_list, dim=1)
            h_slow_final = h_slow_t

        # === Combine paths ===
        out_fast = self.to_out_fast(h_fast_seq)
        out_slow = self.ema_out_proj(h_slow_seq)

        out = out_fast + out_slow

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast_final, h_slow_final], dim=-1)
            return out, combined_hidden
        return out


if __name__ == "__main__":
    print("Testing FlashGRU_EMA...")

    if not FLASHRNN_AVAILABLE:
        print("FlashRNN not installed. Skipping test.")
        exit(0)

    model = FlashGRU_EMA(
        dim=256,
        expansion_factor=1.0,
        ema_dim=256,
        ema_alpha=0.01
    ).cuda().bfloat16()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)

    out, hidden = model(x, return_next_prev_hidden=True)
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"EMA alpha: {model.alpha.item():.4f}")

    # Benchmark
    import time
    model.eval()
    x_bench = torch.randn(32, 512, 256, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x_bench)
    torch.cuda.synchronize()

    # Time it
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(x_bench)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = 32 * 512 * 10
    print(f"Throughput: {tokens / elapsed:,.0f} tok/s")
    print("Done!")

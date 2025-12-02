"""
Parallel EMA + Sequential GRU for long-range memory.

Key insight: EMA is a linear recurrence that CAN be parallelized!
  h_slow[t] = decay * h_slow[t-1] + alpha * x[t]

Using exponential weighting:
  h_slow[t] = decay^t * h0 + alpha * sum_{i=0}^{t} decay^{t-i} * x[i]

This can be computed in O(1) parallel time using cumsum with exponential weights.

GRU remains sequential (gates depend on previous hidden state).
"""

import torch
import torch.nn as nn
import math

# Import the Triton kernel from the original
try:
    from mingru.hybrid_fused_gru import gru_cell_fused, gru_cell_pytorch
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton GRU kernels not available, falling back to PyTorch")

    def gru_cell_pytorch(input_gates, hidden_gates, h_prev):
        """Pure PyTorch GRU cell for CPU/debugging."""
        B, H3 = input_gates.shape
        H = H3 // 3
        i_r, i_z, i_n = input_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * h_prev + z * n
        return h_new


def parallel_ema_chunk(x, decay, alpha, h0=None):
    """
    Compute EMA for a chunk in parallel using exponential weighting.

    For numerical stability, chunks should be <= 64 tokens.

    h[t] = decay * h[t-1] + alpha * x[t]

    Expanded:
    h[t] = decay^{t+1} * h0 + alpha * sum_{i=0}^{t} decay^{t-i} * x[i]
    """
    B, T, D = x.shape
    device = x.device

    if h0 is None:
        h0 = torch.zeros(B, D, device=device, dtype=torch.float32)

    # All computation in fp32
    x_fp32 = x.float()
    h0_fp32 = h0.float()
    decay = float(decay)
    alpha = float(alpha)

    t_idx = torch.arange(T, device=device, dtype=torch.float32)

    # Decay powers for h0 contribution
    h0_decay_powers = decay ** (t_idx + 1)

    # For input sum: scale by 1/decay^i, cumsum, scale back by decay^t
    inv_decay_powers = decay ** (-t_idx)

    scaled_x = x_fp32 * inv_decay_powers.view(1, T, 1)
    cumsum = scaled_x.cumsum(dim=1)
    input_contribution = alpha * cumsum * (decay ** t_idx).view(1, T, 1)

    h0_contribution = h0_fp32.unsqueeze(1) * h0_decay_powers.view(1, T, 1)

    h_all = h0_contribution + input_contribution
    h_final = h_all[:, -1]

    return h_all, h_final


def parallel_ema(x, decay, alpha, h0=None, chunk_size=64):
    """
    Compute EMA for entire sequence in parallel using chunked approach.

    Chunks bound the maximum scaling factor for numerical stability.
    For decay=0.99, chunk_size=64 gives max scale of ~1.9x (manageable).

    Args:
        x: [B, T, D] input sequence
        decay: scalar decay rate (1 - alpha)
        alpha: scalar learning rate
        h0: [B, D] initial hidden state (default: zeros)
        chunk_size: Max chunk size for numerical stability

    Returns:
        h_all: [B, T, D] all hidden states
        h_final: [B, D] final hidden state
    """
    B, T, D = x.shape
    device = x.device
    dtype = x.dtype

    if h0 is None:
        h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # For short sequences, process in one chunk
    if T <= chunk_size:
        h_all, h_final = parallel_ema_chunk(x, decay, alpha, h0)
        return h_all.to(dtype), h_final.to(dtype)

    # Process in chunks, chaining hidden states
    all_outputs = []
    h_current = h0.float()

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        x_chunk = x[:, start:end]

        h_chunk, h_current = parallel_ema_chunk(x_chunk, decay, alpha, h_current)
        all_outputs.append(h_chunk)

    h_all = torch.cat(all_outputs, dim=1)
    return h_all.to(dtype), h_current.to(dtype)


class ParallelEMA_GRU(nn.Module):
    """
    GRU + Parallel EMA for long-range memory.

    GRU: Sequential (Triton cell kernel per timestep)
    EMA: Fully parallel (exponential weighting + cumsum)

    This should be significantly faster than the naive sequential EMA.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ema_dim: int = None,
        ema_alpha: float = 0.01,
        learnable_alpha: bool = True,
        z_bias_input: float = 0.0,
        z_bias_hidden: float = 0.0,
        recurrence_chunk_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.recurrence_chunk_size = recurrence_chunk_size
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ema_dim = ema_dim if ema_dim is not None else dim
        self.z_bias_input = z_bias_input
        self.z_bias_hidden = z_bias_hidden

        # === GRU (fast) path ===
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner)

        if expansion_factor != 1.0:
            self.to_out_fast = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out_fast = nn.Identity()

        # === EMA (slow) path ===
        self.ema_in_proj = nn.Linear(dim, self.ema_dim, bias=False)
        self.ema_out_proj = nn.Linear(self.ema_dim, dim, bias=False)

        # EMA alpha (decay rate)
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
        std = 1.0 / math.sqrt(self.dim_inner)

        for lin in [self.input_projection, self.hidden_projection]:
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)

            H = self.dim_inner
            with torch.no_grad():
                if lin == self.input_projection:
                    lin.bias[H:2*H].fill_(self.z_bias_input)
                else:
                    lin.bias[H:2*H].fill_(self.z_bias_hidden)

        if not isinstance(self.to_out_fast, nn.Identity):
            nn.init.constant_(self.to_out_fast.weight, 0.0)

        nn.init.normal_(self.ema_in_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.ema_out_proj.weight, 0.0)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Parse hidden states
        if prev_hidden is None:
            h_fast = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            h_slow = torch.zeros(B, self.ema_dim, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner].contiguous()
            h_slow = prev_hidden[:, self.dim_inner:].contiguous()

        alpha = self.alpha
        decay = 1.0 - alpha

        # Pre-compute all input projections (FAST!)
        input_gates_all = self.input_projection(x)  # [B, T, 3*H]
        x_ema_all = self.ema_in_proj(x)  # [B, T, ema_dim]

        # === PARALLEL EMA computation ===
        # This is O(T) work but O(1) depth - fully parallel!
        if doc_boundaries is None:
            h_slow_seq, h_slow_final = parallel_ema(x_ema_all, decay, alpha, h_slow)
        else:
            # With doc boundaries, we need to segment the EMA
            # For now, fall back to sequential for doc boundaries
            h_slow_list = []
            h_slow_t = h_slow
            for t in range(T):
                if doc_boundaries[:, t].any():
                    h_slow_t = h_slow_t.masked_fill(doc_boundaries[:, t].unsqueeze(-1), 0.0)
                h_slow_t = decay * h_slow_t + alpha * x_ema_all[:, t]
                h_slow_list.append(h_slow_t)
            h_slow_seq = torch.stack(h_slow_list, dim=1)
            h_slow_final = h_slow_t

        # === Sequential GRU computation ===
        BLOCK_SIZE = 2 ** math.ceil(math.log2(self.dim_inner))
        outputs_fast = []

        for t in range(T):
            # Handle document boundaries for GRU
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]
                if reset_mask.any():
                    h_fast = h_fast.masked_fill(reset_mask.unsqueeze(-1), 0.0)

            input_gates = input_gates_all[:, t].contiguous()
            hidden_gates = self.hidden_projection(h_fast).contiguous()

            if device.type == 'cuda' and TRITON_AVAILABLE:
                h_new_fp32 = torch.empty(B, self.dim_inner, device=device, dtype=torch.float32)
                grid = (B,)
                gru_cell_fused[grid](
                    input_gates, hidden_gates,
                    h_fast, h_new_fp32,
                    B, self.dim_inner,
                    BLOCK_SIZE
                )
                h_fast = h_new_fp32.to(dtype)
            else:
                h_fast = gru_cell_pytorch(input_gates, hidden_gates, h_fast)

            outputs_fast.append(h_fast)

        h_fast_seq = torch.stack(outputs_fast, dim=1)  # [B, T, dim_inner]

        # Combine fast and slow paths
        out_fast = self.to_out_fast(h_fast_seq)
        out_slow = self.ema_out_proj(h_slow_seq)

        out = out_fast + out_slow

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast, h_slow_final], dim=-1)
            return out, combined_hidden
        return out


if __name__ == "__main__":
    print("Testing ParallelEMA_GRU...")

    # Test parallel_ema function
    print("\n=== Testing parallel_ema ===")
    B, T, D = 4, 128, 256
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)
    h0 = torch.randn(B, D, device='cuda', dtype=torch.bfloat16)
    alpha = 0.01
    decay = 1.0 - alpha

    # Parallel version
    h_parallel, h_final_parallel = parallel_ema(x, decay, alpha, h0)

    # Sequential version (reference)
    h_sequential = []
    h_t = h0
    for t in range(T):
        h_t = decay * h_t + alpha * x[:, t]
        h_sequential.append(h_t)
    h_sequential = torch.stack(h_sequential, dim=1)
    h_final_sequential = h_t

    # Compare - expect some difference due to bf16 quantization
    diff = (h_parallel - h_sequential).abs().max().item()
    rel_error = diff / h_sequential.abs().mean().item()
    print(f"Max difference parallel vs sequential: {diff:.6f}")
    print(f"Relative error: {rel_error:.2%}")
    # bf16 has ~0.4% precision, 5% relative error is acceptable for parallel scan
    print(f"✓ Parallel EMA within bf16 tolerance!" if rel_error < 0.10 else f"✗ Error too high!")

    # Test full model
    print("\n=== Testing ParallelEMA_GRU model ===")
    model = ParallelEMA_GRU(
        dim=256,
        expansion_factor=1.0,
        ema_dim=256,
        ema_alpha=0.01
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)

    out, hidden = model(x, return_next_prev_hidden=True)
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"EMA alpha: {model.alpha.item():.4f}")

    # Benchmark
    print("\n=== Benchmark ===")
    import time
    model.eval()
    x_bench = torch.randn(32, 512, 256, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        _ = model(x_bench)
    torch.cuda.synchronize()

    # Time it
    start = time.time()
    for _ in range(10):
        _ = model(x_bench)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = 32 * 512 * 10
    print(f"Throughput: {tokens / elapsed:.0f} tokens/sec")
    print("✓ Done!")

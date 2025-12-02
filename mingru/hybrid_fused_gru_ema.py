"""
HybridFusedGRU with EMA (Exponential Moving Average) path for long-range memory.

Key insight: GRU forgets quickly (z ~ 0.5 means half-life of ~1 token).
EMA with small alpha (0.01) has half-life of ~70 tokens, preserving long-range info.

Architecture:
  h_fast = GRU(x, h_fast_prev)           # Fast, local patterns (Triton kernel)
  h_slow = (1-alpha)*h_slow_prev + alpha*x  # Slow, long-range memory
  output = W_fast @ h_fast + W_slow @ h_slow

Uses the same memory-efficient approach as HybridFusedGRU (Triton cell kernels).
"""

import torch
import torch.nn as nn
import math

# Import the Triton kernel from the original
try:
    from mingru.hybrid_fused_gru import gru_cell_fused
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton GRU kernels not available, falling back to PyTorch")


def gru_cell_pytorch(input_gates, hidden_gates, h_prev):
    """Pure PyTorch GRU cell for CPU/debugging."""
    B, H3 = input_gates.shape
    H = H3 // 3

    # Split gates
    i_r, i_z, i_n = input_gates.chunk(3, dim=-1)
    h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)

    # GRU computation
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h_new = (1 - z) * h_prev + z * n

    return h_new


class HybridFusedGRU_EMA(nn.Module):
    """
    Hybrid GRU with parallel EMA path for long-range memory.

    Uses the memory-efficient HybridFusedGRU approach (Triton kernels)
    plus an EMA path for long-range context.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ema_dim: int = None,  # Dimension for EMA state (default: dim)
        ema_alpha: float = 0.01,  # EMA decay rate (small = longer memory)
        learnable_alpha: bool = True,  # Make alpha learnable
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

        # === GRU (fast) path - same as HybridFusedGRU ===
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner)

        # Output projection for GRU
        if expansion_factor != 1.0:
            self.to_out_fast = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out_fast = nn.Identity()

        # === EMA (slow) path ===
        # Project input to EMA dimension
        self.ema_in_proj = nn.Linear(dim, self.ema_dim, bias=False)
        # Project EMA to output
        self.ema_out_proj = nn.Linear(self.ema_dim, dim, bias=False)

        # EMA alpha (decay rate)
        if learnable_alpha:
            # Initialize in logit space for stable learning
            # alpha = sigmoid(alpha_logit), init to ~0.01
            init_logit = math.log(ema_alpha / (1 - ema_alpha))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('alpha_logit', torch.tensor(math.log(ema_alpha / (1 - ema_alpha))))

        self._init_weights()

    @property
    def alpha(self):
        """Get EMA alpha from logit (ensures 0 < alpha < 1)"""
        return torch.sigmoid(self.alpha_logit)

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.dim_inner)

        # GRU weights - same as HybridFusedGRU
        for lin in [self.input_projection, self.hidden_projection]:
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)

            H = self.dim_inner
            with torch.no_grad():
                if lin == self.input_projection:
                    lin.bias[H:2*H].fill_(self.z_bias_input)
                else:
                    lin.bias[H:2*H].fill_(self.z_bias_hidden)

        # GRU output projection - zero init for residual
        if not isinstance(self.to_out_fast, nn.Identity):
            nn.init.constant_(self.to_out_fast.weight, 0.0)

        # EMA projections - small init
        nn.init.normal_(self.ema_in_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.ema_out_proj.weight, 0.0)  # Zero init so EMA starts inactive

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Hidden state is concatenated: [h_fast | h_slow]
        if prev_hidden is None:
            h_fast = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            h_slow = torch.zeros(B, self.ema_dim, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner]
            h_slow = prev_hidden[:, self.dim_inner:]

        # Get EMA alpha and decay
        alpha = self.alpha
        decay = 1.0 - alpha

        # Pre-compute ALL input projections at once (FAST!)
        input_gates_all = self.input_projection(x)  # [B, T, 3*H]
        x_ema_all = self.ema_in_proj(x)  # [B, T, ema_dim]

        outputs_fast = []
        outputs_slow = []

        # Find next power of 2 >= dim_inner for Triton
        BLOCK_SIZE = 2 ** math.ceil(math.log2(self.dim_inner))

        # Process each timestep
        for t in range(T):
            # Handle document boundaries (reset states)
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]
                if reset_mask.any():
                    h_fast = h_fast.masked_fill(reset_mask.unsqueeze(-1), 0.0)
                    h_slow = h_slow.masked_fill(reset_mask.unsqueeze(-1), 0.0)

            # === GRU (fast) path ===
            input_gates = input_gates_all[:, t].contiguous()
            hidden_gates = self.hidden_projection(h_fast).contiguous()

            if device.type == 'cuda' and TRITON_AVAILABLE:
                # Use Triton kernel
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
                # CPU fallback
                h_fast = gru_cell_pytorch(input_gates, hidden_gates, h_fast)

            # === EMA (slow) path - simple and fast ===
            x_ema_t = x_ema_all[:, t]
            h_slow = decay * h_slow + alpha * x_ema_t

            outputs_fast.append(h_fast)
            outputs_slow.append(h_slow)

        # Stack outputs
        h_fast_seq = torch.stack(outputs_fast, dim=1)  # [B, T, dim_inner]
        h_slow_seq = torch.stack(outputs_slow, dim=1)  # [B, T, ema_dim]

        # Combine fast and slow paths
        out_fast = self.to_out_fast(h_fast_seq)  # [B, T, D]
        out_slow = self.ema_out_proj(h_slow_seq)  # [B, T, D]

        out = out_fast + out_slow

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast, h_slow], dim=-1)
            return out, combined_hidden
        return out


# Test code
if __name__ == "__main__":
    print("Testing HybridFusedGRU_EMA (Triton GRU + EMA)...")

    model = HybridFusedGRU_EMA(
        dim=256,
        expansion_factor=1.0,
        ema_dim=256,
        ema_alpha=0.01,
        learnable_alpha=True
    ).cuda()

    x = torch.randn(2, 128, 256).cuda()

    # First forward
    out1, hidden = model(x, return_next_prev_hidden=True)
    print(f"Output shape: {out1.shape}")
    print(f"Hidden state (concatenated): {hidden.shape}")
    print(f"  - Fast path dim: {model.dim_inner}, Slow path dim: {model.ema_dim}")
    print(f"EMA alpha: {model.alpha.item():.4f}")

    # Continue with hidden state
    x2 = torch.randn(2, 128, 256).cuda()
    out2, hidden2 = model(x2, prev_hidden=hidden, return_next_prev_hidden=True)
    print(f"Continued output shape: {out2.shape}")
    print(f"Continued hidden state: {hidden2.shape}")

    # Benchmark
    import time
    model.eval()
    x_bench = torch.randn(32, 512, 256).cuda()

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
    print(f"Throughput: {tokens / elapsed:.0f} tokens/sec (single GPU, inference)")

    print("✓ Test passed!")

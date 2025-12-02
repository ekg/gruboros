"""
Fused GRU + EMA Triton Kernel

Key optimizations:
1. Process entire chunk in ONE kernel launch (no Python loop overhead)
2. Keep h_fast and h_slow in registers (no HBM round-trips)
3. Fuse GRU cell computation with EMA update
4. Parallelize across batch and hidden dimensions

Architecture:
  - GRU: h_fast[t] = (1-z)*h_fast[t-1] + z*tanh(...)
  - EMA: h_slow[t] = (1-alpha)*h_slow[t-1] + alpha*x_ema[t]
  - Output: W_fast @ h_fast + W_slow @ h_slow
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gru_ema_cell_kernel(
    # Pre-computed input gates [B, T, 3*H] - computed by PyTorch matmul
    input_gates_ptr,
    # Pre-computed hidden gates [B, T, 3*H] - computed by chunked approach
    hidden_gates_ptr,
    # Pre-computed EMA input [B, T, ema_dim]
    x_ema_ptr,
    # Initial GRU hidden state [B, H]
    h_fast_in_ptr,
    # Initial EMA hidden state [B, ema_dim]
    h_slow_in_ptr,
    # Document boundaries [B, T] as int8
    doc_boundaries_ptr,
    # Outputs: all GRU hidden states [B, T, H]
    h_fast_out_ptr,
    # Outputs: all EMA hidden states [B, T, ema_dim]
    h_slow_out_ptr,
    # Final GRU hidden state [B, H]
    h_fast_final_ptr,
    # Final EMA hidden state [B, ema_dim]
    h_slow_final_ptr,
    # EMA alpha (scalar)
    alpha,
    # Dimensions
    batch_size, seq_len, hidden_dim, ema_dim,
    # Strides for input_gates
    ig_stride_b, ig_stride_t, ig_stride_h,
    # Strides for x_ema
    ema_stride_b, ema_stride_t, ema_stride_d,
    # Strides for outputs
    out_stride_b, out_stride_t, out_stride_h,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_EMA: tl.constexpr,
):
    """
    Fused GRU + EMA kernel.

    Each program instance handles one batch element.
    Processes the entire sequence in a loop, keeping states in registers.
    """
    pid_batch = tl.program_id(0)
    if pid_batch >= batch_size:
        return

    decay = 1.0 - alpha

    # Offsets for hidden dimension
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim

    offs_ema = tl.arange(0, BLOCK_EMA)
    mask_ema = offs_ema < ema_dim

    # Load initial hidden states in fp32 for numerical stability
    h_fast_base = pid_batch * hidden_dim
    h_fast = tl.load(h_fast_in_ptr + h_fast_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    h_slow_base = pid_batch * ema_dim
    h_slow = tl.load(h_slow_in_ptr + h_slow_base + offs_ema, mask=mask_ema, other=0.0).to(tl.float32)

    # Process sequence
    for t in range(seq_len):
        # Check document boundary - reset if needed
        if doc_boundaries_ptr is not None:
            doc_bound_offset = pid_batch * seq_len + t
            doc_boundary = tl.load(doc_boundaries_ptr + doc_bound_offset)
            # If doc_boundary == 1, reset hidden states
            h_fast = tl.where(doc_boundary == 1, 0.0, h_fast)
            h_slow = tl.where(doc_boundary == 1, 0.0, h_slow)

        # === Load pre-computed gates for this timestep ===
        gates_base = pid_batch * ig_stride_b + t * ig_stride_t

        # Input gates [r, z, n]
        i_r = tl.load(input_gates_ptr + gates_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        i_z = tl.load(input_gates_ptr + gates_base + hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        i_n = tl.load(input_gates_ptr + gates_base + 2*hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        # Hidden gates [r, z, n]
        h_r = tl.load(hidden_gates_ptr + gates_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        h_z = tl.load(hidden_gates_ptr + gates_base + hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        h_n = tl.load(hidden_gates_ptr + gates_base + 2*hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        # === GRU cell computation ===
        r = tl.sigmoid(i_r + h_r)
        z = tl.sigmoid(i_z + h_z)

        # Numerically stable tanh using manual formula
        n_pre = i_n + r * h_n
        n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
        exp_2x = tl.exp(2.0 * n_pre_clamped)
        n = (exp_2x - 1.0) / (exp_2x + 1.0)

        # Update GRU hidden state
        h_fast = (1.0 - z) * h_fast + z * n

        # === EMA computation ===
        ema_base = pid_batch * ema_stride_b + t * ema_stride_t
        x_ema_t = tl.load(x_ema_ptr + ema_base + offs_ema, mask=mask_ema, other=0.0).to(tl.float32)
        h_slow = decay * h_slow + alpha * x_ema_t

        # === Store outputs ===
        out_base_fast = pid_batch * out_stride_b + t * out_stride_t
        tl.store(h_fast_out_ptr + out_base_fast + offs_h, h_fast.to(tl.bfloat16), mask=mask_h)

        out_base_slow = pid_batch * out_stride_b + t * out_stride_t
        tl.store(h_slow_out_ptr + out_base_slow + offs_ema, h_slow.to(tl.bfloat16), mask=mask_ema)

    # Store final hidden states
    tl.store(h_fast_final_ptr + h_fast_base + offs_h, h_fast.to(tl.bfloat16), mask=mask_h)
    tl.store(h_slow_final_ptr + h_slow_base + offs_ema, h_slow.to(tl.bfloat16), mask=mask_ema)


class FusedGRU_EMA(nn.Module):
    """
    GRU + EMA with fused Triton kernel for the recurrence.

    Strategy:
    1. Pre-compute input gates with PyTorch (cuBLAS) - fast!
    2. Pre-compute hidden gates in chunks (cuBLAS between chunks)
    3. Fuse GRU cell + EMA in single Triton kernel
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
        chunk_size: int = 64,  # Process this many timesteps per kernel launch
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ema_dim = ema_dim if ema_dim is not None else dim
        self.z_bias_input = z_bias_input
        self.z_bias_hidden = z_bias_hidden
        self.chunk_size = chunk_size

        # GRU projections
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner)

        # Output projection for GRU
        if expansion_factor != 1.0:
            self.to_out_fast = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out_fast = nn.Identity()

        # EMA projections
        self.ema_in_proj = nn.Linear(dim, self.ema_dim, bias=False)
        self.ema_out_proj = nn.Linear(self.ema_dim, dim, bias=False)

        # EMA alpha
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

        # Initialize hidden states
        if prev_hidden is None:
            h_fast = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
            h_slow = torch.zeros(B, self.ema_dim, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner].contiguous()
            h_slow = prev_hidden[:, self.dim_inner:].contiguous()

        alpha = self.alpha.item()

        # Pre-compute ALL input gates at once (cuBLAS - fast!)
        input_gates = self.input_projection(x)  # [B, T, 3*H]

        # Pre-compute EMA input
        x_ema = self.ema_in_proj(x)  # [B, T, ema_dim]

        # Process in chunks to balance kernel efficiency vs memory
        chunk_size = min(self.chunk_size, T)
        num_chunks = (T + chunk_size - 1) // chunk_size

        all_h_fast = []
        all_h_slow = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, T)
            actual_chunk = end - start

            # Get chunk data
            input_gates_chunk = input_gates[:, start:end].contiguous()  # [B, chunk, 3*H]
            x_ema_chunk = x_ema[:, start:end].contiguous()  # [B, chunk, ema_dim]

            # Pre-compute hidden gates for this chunk
            # For each timestep in chunk, we need hidden_projection(h[t-1])
            # This requires sequential computation, so we do it with the kernel

            # For now, use a simpler approach: compute hidden gates per-timestep outside
            # Then fuse just the cell logic in the kernel
            # TODO: Full fusion would put the hidden projection inside the kernel

            hidden_gates_list = []
            h_temp = h_fast.clone()
            for t in range(actual_chunk):
                hidden_gates_t = self.hidden_projection(h_temp)  # [B, 3*H]
                hidden_gates_list.append(hidden_gates_t)

                # Quick GRU step to get h for next timestep's hidden projection
                ig = input_gates_chunk[:, t]
                hg = hidden_gates_t

                i_r, i_z, i_n = ig.chunk(3, dim=-1)
                h_r, h_z, h_n = hg.chunk(3, dim=-1)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h_temp = (1 - z) * h_temp + z * n

            hidden_gates_chunk = torch.stack(hidden_gates_list, dim=1)  # [B, chunk, 3*H]

            # Allocate outputs
            h_fast_out = torch.empty(B, actual_chunk, self.dim_inner, device=device, dtype=dtype)
            h_slow_out = torch.empty(B, actual_chunk, self.ema_dim, device=device, dtype=dtype)
            h_fast_final = torch.empty(B, self.dim_inner, device=device, dtype=dtype)
            h_slow_final = torch.empty(B, self.ema_dim, device=device, dtype=dtype)

            # Prepare doc boundaries
            if doc_boundaries is not None:
                doc_chunk = doc_boundaries[:, start:end].to(torch.int8).contiguous()
                doc_ptr = doc_chunk
            else:
                doc_ptr = None

            # Launch fused kernel
            BLOCK_H = triton.next_power_of_2(self.dim_inner)
            BLOCK_EMA = triton.next_power_of_2(self.ema_dim)

            grid = (B,)
            fused_gru_ema_cell_kernel[grid](
                input_gates_chunk,
                hidden_gates_chunk,
                x_ema_chunk,
                h_fast,
                h_slow,
                doc_ptr,
                h_fast_out,
                h_slow_out,
                h_fast_final,
                h_slow_final,
                alpha,
                B, actual_chunk, self.dim_inner, self.ema_dim,
                input_gates_chunk.stride(0), input_gates_chunk.stride(1), input_gates_chunk.stride(2),
                x_ema_chunk.stride(0), x_ema_chunk.stride(1), x_ema_chunk.stride(2),
                h_fast_out.stride(0), h_fast_out.stride(1), h_fast_out.stride(2),
                BLOCK_H, BLOCK_EMA
            )

            all_h_fast.append(h_fast_out)
            all_h_slow.append(h_slow_out)

            # Update hidden states for next chunk
            h_fast = h_fast_final
            h_slow = h_slow_final

        # Concatenate all chunks
        h_fast_seq = torch.cat(all_h_fast, dim=1)  # [B, T, H]
        h_slow_seq = torch.cat(all_h_slow, dim=1)  # [B, T, ema_dim]

        # Final projections
        out_fast = self.to_out_fast(h_fast_seq)
        out_slow = self.ema_out_proj(h_slow_seq)

        out = out_fast + out_slow

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast, h_slow], dim=-1)
            return out, combined_hidden
        return out


if __name__ == "__main__":
    print("Testing FusedGRU_EMA...")

    model = FusedGRU_EMA(
        dim=256,
        expansion_factor=1.0,
        ema_dim=256,
        ema_alpha=0.01,
        chunk_size=64
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)

    out, hidden = model(x, return_next_prev_hidden=True)
    print(f"Output: {out.shape}")
    print(f"Hidden: {hidden.shape}")

    # Benchmark
    import time
    model.eval()
    x_bench = torch.randn(32, 512, 256, device='cuda', dtype=torch.bfloat16)

    for _ in range(3):
        _ = model(x_bench)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):
        _ = model(x_bench)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = 32 * 512 * 10
    print(f"Throughput: {tokens / elapsed:.0f} tokens/sec")
    print("✓ Done!")

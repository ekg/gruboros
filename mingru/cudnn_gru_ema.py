"""
cuDNN GRU + EMA: Stable GRU via cuDNN + Long-range memory via parallel EMA.

This is a DDP-compatible alternative to FlashGRU_EMA.
FlashRNN has bugs with DDP (bias gradient kernel crashes), so we use
PyTorch's cuDNN-backed nn.GRU instead.

Architecture:
  - h_fast = cuDNN_GRU(x)        # Stable, DDP-compatible
  - h_slow = parallel_EMA(x)     # O(1) depth for long-range context
  - output = W_fast @ h_fast + W_slow @ h_slow
"""

import torch
import torch.nn as nn
import math


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


class CuDNNGRU_EMA(nn.Module):
    """
    cuDNN GRU + Parallel EMA for stable DDP training with long-range memory.

    - GRU path: PyTorch nn.GRU (cuDNN backend) - stable, DDP-compatible
    - EMA path: Parallel cumsum for O(1) depth long-range memory

    This is a drop-in replacement for FlashGRU_EMA that works with DDP.
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

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ema_dim = ema_dim if ema_dim is not None else dim
        self.ema_chunk_size = recurrence_chunk_size

        # === cuDNN GRU (fast path) ===
        # Use PyTorch's nn.GRU which uses cuDNN under the hood
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

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
        dtype = x.dtype

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

        # === cuDNN GRU (fast path) ===
        # IMPORTANT: cuDNN GRU doesn't support bfloat16, must use float32 or float16
        # Cast to float32 for stability, then back to original dtype
        x_gru = x.float().contiguous()

        # Prepare initial hidden state: [1, B, dim_inner]
        if h_fast is not None:
            h0 = h_fast.float().unsqueeze(0).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=torch.float32)

        # Run cuDNN GRU (in float32)
        h_fast_seq, h_fast_final = self.gru(x_gru.to(self.gru.weight_ih_l0.dtype),
                                             h0.to(self.gru.weight_ih_l0.dtype))
        # h_fast_seq: [B, T, dim_inner]
        # h_fast_final: [1, B, dim_inner]
        h_fast_seq = h_fast_seq.to(dtype)
        h_fast_final = h_fast_final.squeeze(0).to(dtype)  # [B, dim_inner]

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
    print("Testing CuDNNGRU_EMA...")

    model = CuDNNGRU_EMA(
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

    # Backward pass test
    loss = out.sum()
    loss.backward()
    print("Backward pass succeeded!")

    print("Done!")

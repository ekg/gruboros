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
        layer_idx: int = None,  # For per-layer alpha initialization
        num_layers: int = None,  # For per-layer alpha initialization
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
        # Per-layer initialization: early layers fast (high α), deep layers slow (low α)
        if layer_idx is not None and num_layers is not None and num_layers > 1:
            # Range from 0.05 (layer 0) to 0.005 (last layer) - exponential decay
            init_alpha = 0.05 * (0.1 ** (layer_idx / (num_layers - 1)))
        else:
            init_alpha = ema_alpha

        if learnable_alpha:
            init_logit = math.log(init_alpha / (1 - init_alpha))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('alpha_logit', torch.tensor(math.log(init_alpha / (1 - init_alpha))))

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


class CuDNNGRU_MultiScaleEMA(nn.Module):
    """
    cuDNN GRU + Multi-Scale Parallel EMA for capturing multiple timescales.

    Instead of a single EMA track, uses multiple EMA tracks at different timescales:
      - Fast EMA (α ~ 0.1): captures short-range patterns
      - Medium EMA (α ~ 0.01): captures medium-range patterns
      - Slow EMA (α ~ 0.001): captures long-range patterns

    All EMAs are computed in parallel, then combined with learned mixing weights.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ema_dim: int = None,
        num_scales: int = 3,
        init_alphas: tuple = (0.1, 0.01, 0.001),
        learnable_alpha: bool = True,
        z_bias_input: float = 0.0,
        z_bias_hidden: float = 0.0,
        recurrence_chunk_size: int = 64,
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ema_dim = ema_dim if ema_dim is not None else dim
        self.ema_chunk_size = recurrence_chunk_size
        self.num_scales = num_scales

        # === cuDNN GRU (fast path) ===
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

        # === Multi-Scale EMA (slow paths) ===
        # Use smaller dim per scale to keep parameter count reasonable
        self.ema_dim_per_scale = self.ema_dim // num_scales

        # Shared input projection for all EMA scales
        self.ema_in_proj = nn.Linear(dim, self.ema_dim_per_scale * num_scales, bias=False)

        # Separate output projections per scale (to allow learned importance)
        self.ema_out_projs = nn.ModuleList([
            nn.Linear(self.ema_dim_per_scale, dim, bias=False)
            for _ in range(num_scales)
        ])

        # Alpha parameters for each scale (learnable in logit space)
        # Optionally adjust based on layer depth
        if layer_idx is not None and num_layers is not None and num_layers > 1:
            # Scale factor: deeper layers have slower EMAs
            depth_factor = 0.3 ** (layer_idx / (num_layers - 1))
            adjusted_alphas = tuple(a * depth_factor for a in init_alphas)
        else:
            adjusted_alphas = init_alphas

        if learnable_alpha:
            self.alpha_logits = nn.ParameterList([
                nn.Parameter(torch.tensor(math.log(a / (1 - a))))
                for a in adjusted_alphas
            ])
        else:
            for i, a in enumerate(adjusted_alphas):
                self.register_buffer(f'alpha_logit_{i}',
                                   torch.tensor(math.log(a / (1 - a))))
            self.alpha_logits = None

        self.learnable_alpha = learnable_alpha
        self._init_weights()

    def get_alphas(self):
        """Get all alpha values."""
        if self.learnable_alpha:
            return [torch.sigmoid(logit) for logit in self.alpha_logits]
        else:
            return [torch.sigmoid(getattr(self, f'alpha_logit_{i}'))
                    for i in range(self.num_scales)]

    def _init_weights(self):
        # GRU output - zero init for residual
        if not isinstance(self.to_out_fast, nn.Identity):
            nn.init.constant_(self.to_out_fast.weight, 0.0)

        # EMA projections - small init, output zero
        nn.init.normal_(self.ema_in_proj.weight, mean=0.0, std=0.01)
        for proj in self.ema_out_projs:
            nn.init.constant_(proj.weight, 0.0)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Calculate hidden size for state management
        ema_total_dim = self.ema_dim_per_scale * self.num_scales

        # Parse hidden states: [h_fast | h_slow_0 | h_slow_1 | ... | h_slow_n]
        if prev_hidden is None:
            h_fast = None
            h_slows = [torch.zeros(B, self.ema_dim_per_scale, device=device, dtype=dtype)
                      for _ in range(self.num_scales)]
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner].contiguous()
            offset = self.dim_inner
            h_slows = []
            for _ in range(self.num_scales):
                h_slows.append(prev_hidden[:, offset:offset + self.ema_dim_per_scale].contiguous())
                offset += self.ema_dim_per_scale

        alphas = self.get_alphas()

        # === cuDNN GRU (fast path) ===
        x_gru = x.float().contiguous()
        if h_fast is not None:
            h0 = h_fast.float().unsqueeze(0).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=torch.float32)

        h_fast_seq, h_fast_final = self.gru(x_gru.to(self.gru.weight_ih_l0.dtype),
                                            h0.to(self.gru.weight_ih_l0.dtype))
        h_fast_seq = h_fast_seq.to(dtype)
        h_fast_final = h_fast_final.squeeze(0).to(dtype)

        # === Multi-Scale Parallel EMA (slow paths) ===
        x_ema = self.ema_in_proj(x)  # [B, T, ema_dim_per_scale * num_scales]

        # Split into separate scale inputs
        x_ema_splits = torch.split(x_ema, self.ema_dim_per_scale, dim=-1)

        h_slow_seqs = []
        h_slow_finals = []

        for i, (x_scale, h_slow, alpha) in enumerate(zip(x_ema_splits, h_slows, alphas)):
            alpha_val = float(alpha.item()) if hasattr(alpha, 'item') else float(alpha)
            decay = 1.0 - alpha_val

            if doc_boundaries is None:
                h_slow_seq, h_slow_final = parallel_ema(
                    x_scale, decay, alpha_val, h_slow, chunk_size=self.ema_chunk_size
                )
            else:
                # With doc boundaries, fall back to sequential
                h_slow_list = []
                h_slow_t = h_slow
                for t in range(T):
                    if doc_boundaries[:, t].any():
                        h_slow_t = h_slow_t.masked_fill(doc_boundaries[:, t].unsqueeze(-1), 0.0)
                    h_slow_t = decay * h_slow_t + alpha_val * x_scale[:, t]
                    h_slow_list.append(h_slow_t)
                h_slow_seq = torch.stack(h_slow_list, dim=1)
                h_slow_final = h_slow_t

            h_slow_seqs.append(h_slow_seq)
            h_slow_finals.append(h_slow_final)

        # === Combine paths ===
        out_fast = self.to_out_fast(h_fast_seq)

        # Sum contributions from all EMA scales
        out_slow = sum(
            proj(h_seq) for proj, h_seq in zip(self.ema_out_projs, h_slow_seqs)
        )

        out = out_fast + out_slow

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast_final] + h_slow_finals, dim=-1)
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

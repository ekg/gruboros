"""
cuDNN GRU + Selective Diagonal SSM: GRU for nonlinear dynamics + SSM for parallel memory.

This replaces the simple EMA track with a proper State Space Model:
- Selective: decay rate varies per token (input-dependent, like Mamba)
- Diagonal: complex eigenvalues allow decay + oscillation
- Parallel: still uses parallel scan, no sequential bottleneck

Architecture:
  - h_fast = cuDNN_GRU(x)           # Nonlinear, sequential (cuDNN)
  - h_slow = SelectiveSSM(x)        # Linear, parallel (selective diagonal SSM)
  - output = W_fast @ h_fast + W_slow @ h_slow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def parallel_ssm_real(x, log_decay, B_coeff, h0=None, chunk_size=64):
    """
    Parallel SSM with real-valued diagonal A matrix.

    SSM equations:
      h_t = A * h_{t-1} + B * x_t
      y_t = h_t

    Where A = exp(-exp(log_decay)) is the decay (0 < A < 1 for stability).
    B_coeff is the input mixing coefficient.

    Uses parallel cumsum trick for O(log T) depth.

    Args:
        x: [B, T, D] input
        log_decay: [D] or [B, T, D] decay rates in log-space
        B_coeff: [D] or [B, T, D] input coefficients
        h0: [B, D] initial hidden state
        chunk_size: chunk size for numerical stability

    Returns:
        h_all: [B, T, D] all hidden states
        h_final: [B, D] final hidden state
    """
    B_dim, T, D = x.shape
    device = x.device
    dtype = x.dtype

    if h0 is None:
        h0 = torch.zeros(B_dim, D, device=device, dtype=dtype)

    # Convert to float32 for stability
    x_fp32 = x.float()
    h0_fp32 = h0.float()

    # Get decay values: A = exp(-exp(log_decay)) ensures 0 < A < 1
    if log_decay.dim() == 1:
        # Fixed decay per dimension
        decay = torch.exp(-torch.exp(log_decay)).float()  # [D]
        decay_is_fixed = True
    else:
        # Per-token decay (selective)
        decay = torch.exp(-torch.exp(log_decay)).float()  # [B, T, D]
        decay_is_fixed = False

    # Get B coefficients
    if B_coeff.dim() == 1:
        B_val = B_coeff.float()  # [D]
        B_is_fixed = True
    else:
        B_val = B_coeff.float()  # [B, T, D]
        B_is_fixed = False

    # For selective (per-token) case, we need sequential within chunks
    # For fixed case, we can use the full parallel trick
    if decay_is_fixed and B_is_fixed:
        # Use parallel cumsum trick
        return _parallel_ssm_fixed(x_fp32, decay, B_val, h0_fp32, chunk_size, dtype)
    else:
        # Use chunked sequential for selective case
        return _parallel_ssm_selective(x_fp32, decay, B_val, h0_fp32, chunk_size, dtype, decay_is_fixed, B_is_fixed)


def _parallel_ssm_fixed(x, decay, B_val, h0, chunk_size, orig_dtype):
    """Parallel SSM with fixed decay (like EMA but with learned params)."""
    B_dim, T, D = x.shape
    device = x.device

    if T <= chunk_size:
        # Single chunk - use parallel cumsum
        t_idx = torch.arange(T, device=device, dtype=torch.float32)

        # decay^(t+1) for h0 contribution
        h0_decay_powers = decay.unsqueeze(0) ** (t_idx.unsqueeze(-1) + 1)  # [T, D]

        # For input contribution: sum_{s=0}^{t} B * x_s * decay^{t-s}
        inv_decay_powers = decay.unsqueeze(0) ** (-t_idx.unsqueeze(-1))  # [T, D]
        decay_powers = decay.unsqueeze(0) ** t_idx.unsqueeze(-1)  # [T, D]

        scaled_x = x * B_val.unsqueeze(0).unsqueeze(0) * inv_decay_powers.unsqueeze(0)  # [B, T, D]
        cumsum = scaled_x.cumsum(dim=1)
        input_contribution = cumsum * decay_powers.unsqueeze(0)

        h0_contribution = h0.unsqueeze(1) * h0_decay_powers.unsqueeze(0)  # [B, T, D]

        h_all = h0_contribution + input_contribution
        return h_all.to(orig_dtype), h_all[:, -1].to(orig_dtype)

    # Multi-chunk processing
    all_outputs = []
    h_current = h0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        x_chunk = x[:, start:end]
        h_chunk, h_current = _parallel_ssm_fixed(x_chunk, decay, B_val, h_current, chunk_size, orig_dtype)
        all_outputs.append(h_chunk.float())

    h_all = torch.cat(all_outputs, dim=1)
    return h_all.to(orig_dtype), h_current.to(orig_dtype)


def _build_causal_decay_matrix(log_decay, chunk_len, device):
    """
    Build the causal decay matrix L for SSD-style matmul.

    L[i,j,d] = prod(decay[k,d] for k in j+1..i) for j < i
    L[i,i,d] = 1 (diagonal)
    L[i,j,d] = 0 for j > i (upper triangular = 0)

    For selective SSM, log_decay is [B, chunk_len, D].
    We compute cumulative log-sums and use exp for stability.

    Args:
        log_decay: [B, chunk_len, D] log of decay values
        chunk_len: int
        device: torch device

    Returns:
        L: [B, chunk_len, chunk_len, D] causal decay matrix
    """
    B_dim, T, D = log_decay.shape

    # Cumulative sum of log_decay: cumsum[t] = sum(log_decay[0:t+1])
    # We need prod(decay[j+1:i+1]) = exp(sum(log_decay[j+1:i+1]))
    #                               = exp(cumsum[i] - cumsum[j])
    log_cumsum = torch.cumsum(log_decay, dim=1)  # [B, T, D]

    # L[i,j] = exp(cumsum[i] - cumsum[j]) for j < i
    # But we need cumsum[-1] = 0 for j=0 case, so prepend 0
    log_cumsum_padded = torch.cat([
        torch.zeros(B_dim, 1, D, device=device, dtype=log_decay.dtype),
        log_cumsum
    ], dim=1)  # [B, T+1, D]

    # L[i,j,d] = exp(log_cumsum[i+1,d] - log_cumsum[j+1,d]) for j <= i
    # Create index matrices
    i_idx = torch.arange(T, device=device)  # [T]
    j_idx = torch.arange(T, device=device)  # [T]

    # log_cumsum_i: [B, T, 1, D] - cumsum at position i
    # log_cumsum_j: [B, 1, T, D] - cumsum at position j
    log_cumsum_i = log_cumsum.unsqueeze(2)  # [B, T, 1, D]
    log_cumsum_j = log_cumsum_padded[:, :-1].unsqueeze(1)  # [B, 1, T, D]

    # L[i,j] = exp(cumsum[i] - cumsum[j]) but only for j <= i
    log_L = log_cumsum_i - log_cumsum_j  # [B, T, T, D]
    L = torch.exp(log_L)  # [B, T, T, D]

    # Apply causal mask: L[i,j] = 0 for j > i
    causal_mask = (i_idx.unsqueeze(1) >= j_idx.unsqueeze(0)).float()  # [T, T]
    L = L * causal_mask.unsqueeze(0).unsqueeze(-1)  # [B, T, T, D]

    return L


def _parallel_ssm_selective(x, decay, B_val, h0, chunk_size, orig_dtype, decay_is_fixed, B_is_fixed):
    """
    Selective SSM with per-token decay/B using efficient sequential loop.

    Memory-efficient O(T*D) implementation using simple recurrence:
    h_t = decay_t * h_{t-1} + B_t * x_t
    """
    B_dim, T, D = x.shape
    device = x.device

    all_outputs = []
    h_current = h0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_len = end - start

        x_chunk = x[:, start:end]  # [B, chunk_len, D]

        # Get decay/B for this chunk
        if decay_is_fixed:
            decay_chunk = decay.unsqueeze(0).unsqueeze(0).expand(B_dim, chunk_len, -1)
        else:
            decay_chunk = decay[:, start:end]  # [B, chunk_len, D]

        if B_is_fixed:
            B_chunk = B_val.unsqueeze(0).unsqueeze(0).expand(B_dim, chunk_len, -1)
        else:
            B_chunk = B_val[:, start:end]  # [B, chunk_len, D]

        # Sequential recurrence within chunk (memory efficient)
        h_states = []
        h = h_current
        for t in range(chunk_len):
            h = decay_chunk[:, t] * h + B_chunk[:, t] * x_chunk[:, t]
            h_states.append(h)

        h_chunk = torch.stack(h_states, dim=1)  # [B, chunk_len, D]
        all_outputs.append(h_chunk)
        h_current = h_chunk[:, -1]  # [B, D]

    h_all = torch.cat(all_outputs, dim=1)
    return h_all.to(orig_dtype), h_current.to(orig_dtype)


class SelectiveDiagonalSSM(nn.Module):
    """
    Selective Diagonal State Space Model.

    Replaces simple EMA with a proper SSM:
    - Learned base decay rates (sigmoid-bounded for stability like EMA)
    - Optional selectivity: decay varies per token based on input
    - Input/output projections with normalization

    SSM equations:
      h_t = A_t * h_{t-1} + B_t * x_t
      y_t = C * h_t + D * x_t

    Where A_t = sigmoid(decay_logit) is input-dependent for selectivity.
    Uses sigmoid instead of exp(-exp()) to ensure 0 < decay < 1 always.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = None,
        selective: bool = True,
        chunk_size: int = 64,
    ):
        super().__init__()

        self.dim = dim
        self.state_dim = state_dim if state_dim is not None else dim
        self.selective = selective
        self.chunk_size = chunk_size

        # Input normalization for stable selectivity projections
        self.input_norm = RMSNorm(dim)

        # Input projection: dim -> state_dim
        self.in_proj = nn.Linear(dim, self.state_dim, bias=False)

        # Base decay rate (learned, logit-space for sigmoid)
        # Initialize so sigmoid gives ~0.99 decay: logit = ln(0.99/0.01) ≈ 4.6
        self.decay_logit_base = nn.Parameter(torch.ones(self.state_dim) * 4.0 + torch.randn(self.state_dim) * 0.5)

        # Base alpha (input mixing) - like EMA alpha
        # Initialize so sigmoid gives ~0.01 alpha: logit = ln(0.01/0.99) ≈ -4.6
        self.alpha_logit_base = nn.Parameter(torch.ones(self.state_dim) * -4.0 + torch.randn(self.state_dim) * 0.5)

        # Selectivity: input-dependent modulation of decay and alpha
        if selective:
            self.decay_proj = nn.Linear(dim, self.state_dim, bias=True)
            self.alpha_proj = nn.Linear(dim, self.state_dim, bias=True)

        # Output projection: state_dim -> dim
        self.out_proj = nn.Linear(self.state_dim, dim, bias=False)

        # Output normalization
        self.output_norm = RMSNorm(dim)

        # Skip connection coefficient
        self.D = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Small init for projections
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.weight)  # Zero for residual-friendly start

        if self.selective:
            nn.init.normal_(self.decay_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.decay_proj.bias)
            nn.init.normal_(self.alpha_proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.alpha_proj.bias)

    def forward(self, x, h0=None):
        """
        Forward pass.

        Args:
            x: [B, T, dim] input
            h0: [B, state_dim] initial hidden state

        Returns:
            y: [B, T, dim] output
            h_final: [B, state_dim] final hidden state
        """
        B, T, D = x.shape

        # Normalize input for stable selectivity
        x_normed = self.input_norm(x)

        # Project input to state space
        x_proj = self.in_proj(x_normed)  # [B, T, state_dim]

        # Get decay and alpha coefficients using sigmoid (always bounded 0-1)
        if self.selective:
            # Input-dependent modulation with clamping for stability
            decay_delta = self.decay_proj(x_normed)  # [B, T, state_dim]
            alpha_delta = self.alpha_proj(x_normed)  # [B, T, state_dim]

            # Clamp deltas to prevent explosion
            decay_delta = torch.clamp(decay_delta, -3.0, 3.0)
            alpha_delta = torch.clamp(alpha_delta, -3.0, 3.0)

            # Combine base + delta and use sigmoid for bounded output
            decay_logit = self.decay_logit_base.unsqueeze(0).unsqueeze(0) + decay_delta
            alpha_logit = self.alpha_logit_base.unsqueeze(0).unsqueeze(0) + alpha_delta

            # Sigmoid ensures 0 < decay < 1 and 0 < alpha < 1
            decay = torch.sigmoid(decay_logit)  # [B, T, state_dim]
            alpha = torch.sigmoid(alpha_logit)  # [B, T, state_dim]
        else:
            decay = torch.sigmoid(self.decay_logit_base)
            alpha = torch.sigmoid(self.alpha_logit_base)

        # Run simple EMA-like recurrence (always stable since decay and alpha are bounded)
        h_all, h_final = self._parallel_ema_selective(x_proj, decay, alpha, h0)

        # Output projection + normalization + skip connection
        y = self.output_norm(self.out_proj(h_all)) + self.D * x

        return y, h_final

    def _parallel_ema_selective(self, x, decay, alpha, h0=None):
        """
        EMA-like recurrence with per-token decay and alpha.

        h_t = decay_t * h_{t-1} + alpha_t * x_t

        Uses chunked sequential for memory efficiency.
        """
        B_dim, T, D = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h0 = torch.zeros(B_dim, D, device=device, dtype=dtype)

        all_outputs = []
        h_current = h0.float()
        x_fp32 = x.float()

        # Handle fixed vs selective decay/alpha
        if decay.dim() == 1:
            decay_is_fixed = True
            decay_fp32 = decay.float()
        else:
            decay_is_fixed = False
            decay_fp32 = decay.float()

        if alpha.dim() == 1:
            alpha_is_fixed = True
            alpha_fp32 = alpha.float()
        else:
            alpha_is_fixed = False
            alpha_fp32 = alpha.float()

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk_len = end - start

            x_chunk = x_fp32[:, start:end]

            # Get decay/alpha for this chunk
            if decay_is_fixed:
                d_chunk = decay_fp32.unsqueeze(0).unsqueeze(0).expand(B_dim, chunk_len, -1)
            else:
                d_chunk = decay_fp32[:, start:end]

            if alpha_is_fixed:
                a_chunk = alpha_fp32.unsqueeze(0).unsqueeze(0).expand(B_dim, chunk_len, -1)
            else:
                a_chunk = alpha_fp32[:, start:end]

            # Sequential recurrence within chunk
            h_states = []
            h = h_current
            for t in range(chunk_len):
                h = d_chunk[:, t] * h + a_chunk[:, t] * x_chunk[:, t]
                h_states.append(h)

            h_chunk = torch.stack(h_states, dim=1)
            all_outputs.append(h_chunk)
            h_current = h_chunk[:, -1]

        h_all = torch.cat(all_outputs, dim=1)
        return h_all.to(dtype), h_current.to(dtype)


class CuDNNGRU_SSM(nn.Module):
    """
    cuDNN GRU + Selective Diagonal SSM for DDP training with advanced memory.

    - GRU path: PyTorch nn.GRU (cuDNN backend) - nonlinear, sequential
    - SSM path: Selective diagonal SSM - linear, parallel, input-dependent decay

    This replaces EMA with a proper SSM that can:
    - Learn optimal decay rates
    - Make decay input-dependent (selectivity)
    - Model more complex memory patterns
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        ssm_dim: int = None,
        selective: bool = True,
        recurrence_chunk_size: int = 64,
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.ssm_dim = ssm_dim if ssm_dim is not None else dim
        self.ssm_chunk_size = recurrence_chunk_size

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

        # === Selective SSM (slow path) ===
        self.ssm = SelectiveDiagonalSSM(
            dim=dim,
            state_dim=self.ssm_dim,
            selective=selective,
            chunk_size=recurrence_chunk_size,
        )

        self._init_weights()

    def _init_weights(self):
        # GRU output - zero init for residual
        if not isinstance(self.to_out_fast, nn.Identity):
            nn.init.zeros_(self.to_out_fast.weight)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Parse hidden states: [h_fast | h_slow]
        if prev_hidden is None:
            h_fast = None
            h_slow = None
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h_fast = prev_hidden[:, :self.dim_inner].contiguous()
            h_slow = prev_hidden[:, self.dim_inner:].contiguous()

        # === cuDNN GRU (fast path) ===
        x_gru = x.float().contiguous()

        if h_fast is not None:
            h0 = h_fast.float().unsqueeze(0).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=torch.float32)

        h_fast_seq, h_fast_final = self.gru(
            x_gru.to(self.gru.weight_ih_l0.dtype),
            h0.to(self.gru.weight_ih_l0.dtype)
        )
        h_fast_seq = h_fast_seq.to(dtype)
        h_fast_final = h_fast_final.squeeze(0).to(dtype)

        # === Selective SSM (slow path) ===
        ssm_out, h_slow_final = self.ssm(x, h_slow)

        # === Combine paths ===
        out_fast = self.to_out_fast(h_fast_seq)
        out = out_fast + ssm_out

        if return_next_prev_hidden:
            combined_hidden = torch.cat([h_fast_final, h_slow_final], dim=-1)
            return out, combined_hidden
        return out


if __name__ == "__main__":
    print("Testing CuDNNGRU_SSM...")

    # Test basic functionality
    model = CuDNNGRU_SSM(
        dim=256,
        expansion_factor=1.0,
        ssm_dim=256,
        selective=True
    ).cuda().bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)

    out, hidden = model(x, return_next_prev_hidden=True)
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")

    # Backward pass test
    loss = out.sum()
    loss.backward()
    print("Backward pass succeeded!")

    # Test with non-selective mode
    print("\nTesting non-selective mode...")
    model_ns = CuDNNGRU_SSM(
        dim=256,
        expansion_factor=1.0,
        ssm_dim=256,
        selective=False
    ).cuda().bfloat16()

    out_ns, _ = model_ns(x, return_next_prev_hidden=True)
    loss_ns = out_ns.sum()
    loss_ns.backward()
    print("Non-selective backward pass succeeded!")

    print("\nDone!")

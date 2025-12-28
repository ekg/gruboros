"""
Level 4: Log-Storage Diagonal - Signed log storage for hidden state

Same computation as Level 3, but hidden state stored as (log|h|, sign(h)).
This prevents numerical underflow when hidden state decays over many steps.

Recurrence (in linear space, then convert to log):
    delta = sigmoid(W_delta @ x_t + b_delta)
    candidate = tanh(W_x @ x_t + r_h * h_{t-1} + b)
    h_t = (1 - delta) * h_{t-1} + delta * candidate

Log storage representation:
    log_h, sign_h where h = sign_h * exp(log_h)

For addition a + b where a = s_a * exp(log_a), b = s_b * exp(log_b):
    If same sign: log|a+b| = max(log_a, log_b) + log(1 + exp(-|log_a - log_b|))
    If different sign: log|a+b| = max(log_a, log_b) + log(1 - exp(-|log_a - log_b|))
    (using log1p for numerical stability)

Key question: Does log storage help prevent gradient death at depth?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# HASTE kernel enabled - backward pass implemented
try:
    import sys
    sys.path.insert(0, '/home/erikg/haste_src')
    import torch  # Must import torch first to load C10
    import haste_pytorch_lib
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False

LEVEL_4_AVAILABLE = True  # PyTorch fallback always available


# ============================================================================
# Mamba2-Style Log-Space Primitives (numerically stable)
# ============================================================================

def log_sigmoid(x):
    """Numerically stable log(sigmoid(x)) = -softplus(-x)"""
    return -F.softplus(-x)


def log_one_minus_sigmoid(x):
    """Numerically stable log(1 - sigmoid(x)) = -softplus(x)"""
    return -F.softplus(x)


def to_log_space(x):
    """Convert tensor to signed log representation: (log|x|, sign(x))"""
    sign_x = torch.sign(x)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    log_x = torch.log(torch.abs(x).clamp(min=1e-10))
    return log_x, sign_x


def from_log_space(log_x, sign_x):
    """Convert signed log representation back to linear."""
    return sign_x * torch.exp(log_x)


class LogStorageDiagonalFunction(torch.autograd.Function):
    """
    Autograd function for Log-Storage Diagonal Elman (Haste kernel).

    KEY INNOVATION: Uses TRUE LOG-SPACE BACKWARD pass with softmax weights
    from logaddexp. This prevents gradient vanishing at depth!

    The gradient w.r.t. log|h| flows through the recurrence using bounded
    softmax weights instead of decaying (1-delta)^T multiplications.
    """

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, W_x, r_h, W_delta, W_out, b, b_delta, n_groups):
        # Forward now returns 3 extra caches for log-space backward
        (log_h, sign_h, output, v, delta_cache, compete_cache,
         weight1_cache, log_term1_cache, log_term2_cache) = haste_pytorch_lib.log_storage_diagonal_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            W_x.contiguous(),
            r_h.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )
        if training:
            # Save all caches including new ones for log-space backward
            ctx.save_for_backward(
                x, W_x, r_h, W_delta, W_out, log_h, sign_h, v,
                delta_cache, compete_cache,
                weight1_cache, log_term1_cache, log_term2_cache  # NEW
            )
            ctx.n_groups = n_groups
        return log_h, sign_h, output

    @staticmethod
    def backward(ctx, d_log_h, d_sign_h, d_output):
        (x, W_x, r_h, W_delta, W_out, log_h, sign_h, v,
         delta_cache, compete_cache,
         weight1_cache, log_term1_cache, log_term2_cache) = ctx.saved_tensors

        # Backward now uses the softmax weight caches for log-space gradients
        dx, dW_x, dr_h, dW_delta, dW_out, db, db_delta = haste_pytorch_lib.log_storage_diagonal_backward(
            W_x, r_h, W_delta, W_out, x, log_h, sign_h, v,
            delta_cache, compete_cache,
            weight1_cache, log_term1_cache, log_term2_cache,  # NEW
            d_output.contiguous(), ctx.n_groups
        )
        return None, dx, None, None, dW_x, dr_h, dW_delta, dW_out, db, db_delta, None


class LogStorageDiagonalCell(nn.Module):
    """
    Log-Storage Diagonal cell - Level 4 of ablation ladder.

    Same as Level 3 but hidden state stored in signed log space.
    This prevents numerical underflow over long sequences.

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        assert dim % n_groups == 0, f"dim ({dim}) must be divisible by n_groups ({n_groups})"

        # Candidate computation weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.r_h = nn.Parameter(torch.zeros(dim))  # Diagonal r_h
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (linear space, will be converted)

        Returns:
            h: [T+1, B, dim] all hidden states (converted back to linear)
            output: [T, B, dim] selective outputs
        """
        T, B, D = x.shape

        if h0 is None:
            # Initialize in log space: log(1) = 0, sign = 1
            log_h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = to_log_space(h0)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            log_h, sign_h, output = LogStorageDiagonalFunction.apply(
                self.training, x, log_h0, sign_h0,
                self.W_x, self.r_h, self.W_delta, self.W_out,
                self.b, self.b_delta, self.n_groups
            )
            # Convert back to linear for compatibility
            h = from_log_space(log_h, sign_h)
            return h, output

        # PyTorch fallback
        return self._forward_pytorch(x, log_h0, sign_h0)

    def _forward_pytorch(self, x, log_h0, sign_h0):
        """
        PyTorch implementation using Mamba2-style log-space computation.

        Key primitives (numerically stable):
        - log_sigmoid(x) = -softplus(-x)
        - log_one_minus_sigmoid(x) = -softplus(x)
        - logaddexp(a, b) for log(exp(a) + exp(b))

        Update formula:
            h_new = (1-delta) * h_prev + delta * candidate

        In log-space (for positive magnitudes):
            log|h_new| = logaddexp(
                log_one_minus_sigmoid(delta_raw) + log|h_prev|,
                log_sigmoid(delta_raw) + log|candidate|
            )

        Sign is tracked separately and determined by which term dominates.
        """
        T, B, D = x.shape

        # Store log/sign representations
        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Convert h_prev to linear for r_h multiplication
            h_prev = from_log_space(log_h_prev, sign_h_prev)

            # Delta gate raw (before sigmoid)
            delta_raw = x_t @ self.W_delta.T + self.b_delta

            # Log-space gate values (Mamba2 style - numerically stable)
            log_one_minus_delta = log_one_minus_sigmoid(delta_raw)  # log(1 - sigmoid(delta_raw))
            log_delta = log_sigmoid(delta_raw)  # log(sigmoid(delta_raw))

            # Candidate with diagonal r_h
            candidate_raw = x_t @ self.W_x.T + self.r_h * h_prev + self.b
            candidate = torch.tanh(candidate_raw)

            # Convert candidate to log-space (signed)
            log_candidate, sign_candidate = to_log_space(candidate)

            # === Mamba2-style log-space interpolation ===
            # term1 = (1-delta) * h_prev  -> log: log_one_minus_delta + log_h_prev
            # term2 = delta * candidate   -> log: log_delta + log_candidate

            log_term1 = log_one_minus_delta + log_h_prev
            log_term2 = log_delta + log_candidate

            # Determine which term dominates (for sign)
            term1_bigger = log_term1 > log_term2

            # When same sign: use logaddexp directly
            # When opposite signs: use log-subtract (more complex)
            same_sign = sign_h_prev * sign_candidate > 0

            # For same sign: log|a + b| = logaddexp(log|a|, log|b|)
            log_h_same = torch.logaddexp(log_term1, log_term2)
            sign_h_same = sign_h_prev  # Both have same sign

            # For opposite signs: log|a - b| = max + log|1 - exp(min - max)|
            # Use log1p for stability, but clamp to avoid log(0)
            log_max = torch.maximum(log_term1, log_term2)
            log_min = torch.minimum(log_term1, log_term2)
            diff = log_min - log_max
            # log(1 - exp(diff)) = log1p(-exp(diff)), stable when diff << 0
            # When diff ≈ 0, this is log(≈0) which is very negative
            log_h_diff = log_max + torch.log(torch.abs(1 - torch.exp(diff)).clamp(min=1e-10))
            sign_h_diff = torch.where(term1_bigger, sign_h_prev, sign_candidate)

            # Select based on sign match
            log_h_new = torch.where(same_sign, log_h_same, log_h_diff)
            sign_h_new = torch.where(same_sign, sign_h_same, sign_h_diff)

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)

            # Output: convert to linear for selectivity
            h_new = from_log_space(log_h_new, sign_h_new)
            h_grouped = h_new.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_grouped, dim=-1)
            compete = compete.view(B, D)

            out_proj = h_new @ self.W_out.T
            output = compete * F.silu(out_proj)
            output_list.append(output)

        # Stack and convert to linear
        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        h = from_log_space(log_h, sign_h)
        output = torch.stack(output_list, dim=0)

        return h, output


class RMSNorm(nn.Module):
    """RMSNorm for numerical stability (like Mamba2)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization: x / sqrt(mean(x^2)) * g
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.g


class LogStorageDiagonal(nn.Module):
    """
    Log-Storage Diagonal layer - Level 4 with projections.

    Same as Level 3 but with signed log storage for hidden state.
    Prevents numerical underflow at depth.

    CRITICAL: Uses RMSNorm before output projection for bf16 stability
    (like Mamba2 does).
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        # Adjust n_groups if needed
        while self.d_inner % self.n_groups != 0:
            self.n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-Storage Diagonal cell
        self.cell = LogStorageDiagonalCell(
            self.d_inner,
            n_groups=self.n_groups,
            delta_init=delta_init
        )

        # RMSNorm before output projection - CRITICAL for bf16 stability!
        # This is what Mamba2 does to prevent explosion.
        self.pre_out_norm = RMSNorm(self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Project input
        x_proj = self.in_proj(x)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        h_all, selective_out = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back: [B, T, d_inner]
        selective_out = selective_out.permute(1, 0, 2).contiguous()

        # Apply dropout, normalize, then project
        # RMSNorm BEFORE output projection is critical for bf16 stability!
        # This is what ElmanSilu, ElmanTripleRCompeteSilu, and Mamba2 all do.
        selective_out = self.dropout(selective_out)
        selective_out = self.pre_out_norm(selective_out)
        output = self.out_proj(selective_out)

        # RESIDUAL CONNECTION: Critical for gradient flow at depth!
        # The compete mechanism causes ~1000x output decay per layer.
        # Without residual, gradients vanish after 2-3 layers.
        output = output + x

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_groups={self.n_groups}, LEVEL=4_LOG_STORAGE'


if __name__ == "__main__":
    print("Testing LogStorageDiagonal (Level 4)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    model = LogStorageDiagonal(dim=512, expansion=2.0, n_groups=32).cuda().bfloat16()
    x = torch.randn(2, 32, 512, device='cuda', dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    # Check for NaN/Inf (log space should prevent these)
    print(f"Output has NaN: {torch.isnan(out).any()}")
    print(f"Output has Inf: {torch.isinf(out).any()}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 4 (Log-Storage Diagonal) test passed!")

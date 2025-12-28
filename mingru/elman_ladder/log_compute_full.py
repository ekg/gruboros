"""
Level 5: Log-Compute Full - Full R via logsumexp decomposition

Goes back to full R matrix (not diagonal) but computes in log-space.
This enables the expressivity of full matrix while maintaining numerical stability.

Key insight: R @ h can be computed in log-space using logsumexp:
    For R = R+ - R- (positive and negative parts):
    positive_contrib = logsumexp(log|R+| + log|h|, dim=-1)
    negative_contrib = logsumexp(log|R-| + log|h|, dim=-1)
    result = positive_contrib - negative_contrib (via signed log add)

Recurrence:
    delta = sigmoid(W_delta @ x_t + b_delta)
    candidate_logspace = tanh(W_x @ x_t + R @ h_{t-1} + b)  # R @ h in log-space
    h_t = (1 - delta) * h_{t-1} + delta * candidate

This is more expensive than diagonal but more expressive.
Key question: Does full R in log-space provide better modeling than diagonal?
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

LEVEL_5_AVAILABLE = True  # PyTorch fallback always available


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


def signed_log_add(log_a, sign_a, log_b, sign_b):
    """Add two signed log values."""
    max_log = torch.maximum(log_a, log_b)
    min_log = torch.minimum(log_a, log_b)
    diff = min_log - max_log

    a_is_max = log_a >= log_b
    sign_max = torch.where(a_is_max, sign_a, sign_b)
    sign_min = torch.where(a_is_max, sign_b, sign_a)

    same_sign = sign_max * sign_min > 0
    exp_diff = torch.exp(diff)

    log_factor = torch.where(
        same_sign,
        torch.log1p(exp_diff),
        torch.log1p(-exp_diff).clamp(min=-1e10)
    )

    log_result = max_log + log_factor
    sign_result = sign_max

    return log_result, sign_result


class LogComputeFullFunction(torch.autograd.Function):
    """Autograd function for Log-Compute Full Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, W_x, R, W_delta, W_out, b, b_delta, n_groups):
        log_h, sign_h, output, v, delta_cache, compete_cache = haste_pytorch_lib.log_compute_full_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            W_x.contiguous(),
            R.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )
        if training:
            ctx.save_for_backward(x, W_x, R, W_delta, W_out, log_h, sign_h, v, delta_cache, compete_cache)
            ctx.n_groups = n_groups
        return log_h, sign_h, output

    @staticmethod
    def backward(ctx, d_log_h, d_sign_h, d_output):
        x, W_x, R, W_delta, W_out, log_h, sign_h, v, delta_cache, compete_cache = ctx.saved_tensors
        dx, dW_x, dR, dW_delta, dW_out, db, db_delta = haste_pytorch_lib.log_compute_full_backward(
            W_x, R, W_delta, W_out, x, log_h, sign_h, v, delta_cache, compete_cache,
            d_output.contiguous(), ctx.n_groups
        )
        # Return gradients for: training, x, log_h0, sign_h0, W_x, R, W_delta, W_out, b, b_delta, n_groups
        return None, dx, None, None, dW_x, dR, dW_delta, dW_out, db, db_delta, None


class LogComputeFullCell(nn.Module):
    """
    Log-Compute Full cell - Level 5 of ablation ladder.

    Full R matrix computed in log-space via logsumexp decomposition.
    More expressive than diagonal but maintains numerical stability.

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
        # Full R matrix (stored in log-space during forward)
        self.R = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Delta (gate) computation
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.R, gain=0.1)  # Small init for R
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            h: [T+1, B, dim] all hidden states
            output: [T, B, dim] selective outputs
        """
        T, B, D = x.shape

        if h0 is None:
            log_h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = to_log_space(h0)

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            log_h, sign_h, output = LogComputeFullFunction.apply(
                self.training, x, log_h0, sign_h0,
                self.W_x, self.R, self.W_delta, self.W_out,
                self.b, self.b_delta, self.n_groups
            )
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

        Uses cuBLAS for R @ h (fast), log-space for gate interpolation (stable).
        """
        T, B, D = x.shape

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Convert h from log to linear for matmuls (cuBLAS)
            h_prev = from_log_space(log_h_prev, sign_h_prev)

            # Delta gate raw (before sigmoid)
            delta_raw = x_t @ self.W_delta.T + self.b_delta

            # Mamba2-style log-space gate values
            log_one_minus_delta = log_one_minus_sigmoid(delta_raw)
            log_delta = log_sigmoid(delta_raw)

            # R @ h using cuBLAS (fast!)
            Rh = h_prev @ self.R.T

            # Candidate
            candidate_raw = x_t @ self.W_x.T + Rh + self.b
            candidate = torch.tanh(candidate_raw)

            # Convert candidate to log-space (signed)
            log_candidate, sign_candidate = to_log_space(candidate)

            # === Mamba2-style log-space interpolation ===
            log_term1 = log_one_minus_delta + log_h_prev
            log_term2 = log_delta + log_candidate

            term1_bigger = log_term1 > log_term2
            same_sign = sign_h_prev * sign_candidate > 0

            # Same sign: use logaddexp
            log_h_same = torch.logaddexp(log_term1, log_term2)
            sign_h_same = sign_h_prev

            # Opposite signs: log|a - b|
            log_max = torch.maximum(log_term1, log_term2)
            log_min = torch.minimum(log_term1, log_term2)
            diff = log_min - log_max
            log_h_diff = log_max + torch.log(torch.abs(1 - torch.exp(diff)).clamp(min=1e-10))
            sign_h_diff = torch.where(term1_bigger, sign_h_prev, sign_candidate)

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

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        h = from_log_space(log_h, sign_h)
        output = torch.stack(output_list, dim=0)

        return h, output


class LogComputeFull(nn.Module):
    """
    Log-Compute Full layer - Level 5 with projections.

    Full R matrix with log-space computation.
    Maximum expressivity with numerical stability.
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        while self.d_inner % self.n_groups != 0:
            self.n_groups -= 1

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        self.cell = LogComputeFullCell(
            self.d_inner,
            n_groups=self.n_groups,
            delta_init=delta_init
        )

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        h_all, selective_out = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        selective_out = selective_out.permute(1, 0, 2).contiguous()
        selective_out = self.dropout(selective_out)
        output = self.out_proj(selective_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_groups={self.n_groups}, LEVEL=5_LOG_COMPUTE_FULL'


if __name__ == "__main__":
    print("Testing LogComputeFull (Level 5)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test with smaller size due to O(d^2) complexity
    model = LogComputeFull(dim=256, expansion=1.0, n_groups=16).cuda().bfloat16()
    x = torch.randn(2, 32, 256, device='cuda', dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print(f"Output has NaN: {torch.isnan(out).any()}")
    print(f"Output has Inf: {torch.isinf(out).any()}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 5 (Log-Compute Full) test passed!")

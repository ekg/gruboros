"""
True Log-Space Backward Pass - Proof of Concept

The key insight: gradients in log-space become ADDITIVE instead of multiplicative.

For recurrence: h_t = (1-δ) * h_{t-1} + δ * candidate
In log-space: log|h_t| = logaddexp(log(1-δ) + log|h_{t-1}|, log(δ) + log|c_t|)

Backward through logaddexp:
If z = logaddexp(a, b)
Then dz/da = exp(a - z) = softmax weight for a
     dz/db = exp(b - z) = softmax weight for b

So gradient flow is controlled by softmax weights, NOT by multiplying small numbers!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sigmoid(x):
    """log(sigmoid(x)) = -softplus(-x)"""
    return -F.softplus(-x)


def log_one_minus_sigmoid(x):
    """log(1 - sigmoid(x)) = -softplus(x)"""
    return -F.softplus(x)


def to_log_space(x):
    """Convert to (log|x|, sign(x))"""
    sign_x = torch.sign(x)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    log_x = torch.log(torch.abs(x).clamp(min=1e-38))
    return log_x, sign_x


def from_log_space(log_x, sign_x):
    """Convert back to linear"""
    return sign_x * torch.exp(log_x)


def signed_logaddexp(log_a, sign_a, log_b, sign_b):
    """
    Compute log|a + b| and sign(a + b) where a = sign_a * exp(log_a), b = sign_b * exp(log_b)

    Returns: (log_result, sign_result)
    """
    # Determine which has larger magnitude
    a_bigger = log_a > log_b
    log_max = torch.where(a_bigger, log_a, log_b)
    log_min = torch.where(a_bigger, log_b, log_a)
    sign_max = torch.where(a_bigger, sign_a, sign_b)
    sign_min = torch.where(a_bigger, sign_b, sign_a)

    # Same sign: log|a+b| = log_max + log(1 + exp(log_min - log_max))
    # Diff sign: log|a-b| = log_max + log(1 - exp(log_min - log_max))
    same_sign = sign_a * sign_b > 0

    diff = log_min - log_max

    # For same sign: log1p(exp(diff))
    # For diff sign: log1p(-exp(diff)) but need to handle when diff ≈ 0
    log_factor = torch.where(
        same_sign,
        torch.log1p(torch.exp(diff)),
        torch.log(torch.abs(1 - torch.exp(diff)).clamp(min=1e-38))
    )

    log_result = log_max + log_factor
    sign_result = torch.where(same_sign, sign_max, sign_max)  # Sign of larger magnitude term

    return log_result, sign_result


class TrueLogSpaceGRU(nn.Module):
    """
    GRU with TRUE log-space backward pass.

    Gradients are computed and accumulated in log-space.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Weights
        self.W_x = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.W_delta = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.r_h = nn.Parameter(torch.zeros(dim))  # Diagonal
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), -2.0))

    def forward(self, x, log_h0=None, sign_h0=None):
        """
        Forward pass storing intermediate values for log-space backward.

        Args:
            x: [T, B, D] input
            log_h0, sign_h0: Initial hidden state in log-space

        Returns:
            log_h: [T+1, B, D] all hidden states (log magnitude)
            sign_h: [T+1, B, D] all hidden state signs
        """
        T, B, D = x.shape

        if log_h0 is None:
            log_h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, D, device=x.device, dtype=x.dtype)

        # Storage for all timesteps
        log_h_list = [log_h0]
        sign_h_list = [sign_h0]

        # Also store intermediates for backward
        self.saved_for_backward = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Delta gate (in log-space)
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            log_delta = log_sigmoid(delta_raw)  # log(δ)
            log_one_minus_delta = log_one_minus_sigmoid(delta_raw)  # log(1-δ)

            # Candidate (need h in linear for r_h multiplication)
            h_prev_linear = from_log_space(log_h_prev, sign_h_prev)
            v = x_t @ self.W_x.T + self.r_h * h_prev_linear + self.b
            candidate = torch.tanh(v)
            log_candidate, sign_candidate = to_log_space(candidate)

            # GRU update in log-space:
            # h_new = (1-δ) * h_prev + δ * candidate
            # log|h_new| = logaddexp(log(1-δ) + log|h_prev|, log(δ) + log|candidate|)

            log_term1 = log_one_minus_delta + log_h_prev  # log|(1-δ) * h_prev|
            log_term2 = log_delta + log_candidate  # log|δ * candidate|

            log_h_new, sign_h_new = signed_logaddexp(
                log_term1, sign_h_prev,
                log_term2, sign_candidate
            )

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)

            # Save for backward
            self.saved_for_backward.append({
                'log_h_prev': log_h_prev,
                'sign_h_prev': sign_h_prev,
                'log_h_new': log_h_new,
                'sign_h_new': sign_h_new,
                'log_term1': log_term1,
                'log_term2': log_term2,
                'log_delta': log_delta,
                'log_one_minus_delta': log_one_minus_delta,
                'delta_raw': delta_raw,
                'v': v,
                'candidate': candidate,
                'x_t': x_t,
                'h_prev_linear': h_prev_linear,
            })

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)

        return log_h, sign_h

    def backward_log_space(self, d_log_h_final, d_sign_h_final=None):
        """
        Backward pass computed IN log-space.

        Instead of d_h (linear gradient), we work with (log|d_log_h|, sign).

        Key insight: For z = logaddexp(a, b)
        dz/da = exp(a - z) = softmax(a, b)[0]

        So gradient flow doesn't involve multiplying by small h values!
        """
        T = len(self.saved_for_backward)

        # Initialize gradient in log-space
        # d_log_h is the gradient of loss w.r.t. log|h|
        d_log_h = d_log_h_final.clone()

        # Accumulate gradients
        d_W_x = torch.zeros_like(self.W_x)
        d_W_delta = torch.zeros_like(self.W_delta)
        d_r_h = torch.zeros_like(self.r_h)
        d_b = torch.zeros_like(self.b)
        d_b_delta = torch.zeros_like(self.b_delta)

        for t in reversed(range(T)):
            saved = self.saved_for_backward[t]

            log_h_new = saved['log_h_new']
            log_term1 = saved['log_term1']
            log_term2 = saved['log_term2']
            log_delta = saved['log_delta']
            log_one_minus_delta = saved['log_one_minus_delta']
            delta_raw = saved['delta_raw']
            v = saved['v']
            candidate = saved['candidate']
            x_t = saved['x_t']
            h_prev_linear = saved['h_prev_linear']
            log_h_prev = saved['log_h_prev']

            # Backward through logaddexp
            # z = logaddexp(a, b) => dz/da = exp(a - z), dz/db = exp(b - z)
            weight_term1 = torch.exp(log_term1 - log_h_new).clamp(max=1.0)  # softmax weight
            weight_term2 = torch.exp(log_term2 - log_h_new).clamp(max=1.0)

            # d_log_term1 = weight_term1 * d_log_h
            # d_log_term2 = weight_term2 * d_log_h
            d_log_term1 = weight_term1 * d_log_h
            d_log_term2 = weight_term2 * d_log_h

            # term1 = log_one_minus_delta + log_h_prev
            # d_log_one_minus_delta = d_log_term1
            # d_log_h_prev = d_log_term1  (this is the KEY - gradient flows through!)
            d_log_h_prev = d_log_term1

            # term2 = log_delta + log_candidate
            # d_log_delta = d_log_term2
            # d_log_candidate = d_log_term2
            d_log_delta = d_log_term2
            d_log_candidate = d_log_term2

            # Backward through log_sigmoid and log_one_minus_sigmoid
            # log_delta = -softplus(-delta_raw)
            # d_delta_raw_from_delta = d_log_delta * sigmoid(delta_raw)
            delta = torch.sigmoid(delta_raw)
            d_delta_raw = d_log_delta * delta - d_log_term1 * (1 - delta)

            # Backward through tanh
            # candidate = tanh(v), log_candidate = log|candidate|
            # d_v = d_log_candidate * (1/|candidate|) * (1 - candidate^2) * sign(candidate)
            # But since log_candidate = log|tanh(v)|:
            # d_v = d_log_candidate * (1 - candidate^2) / candidate (when candidate != 0)
            dtanh = 1 - candidate ** 2
            d_v = d_log_candidate * dtanh / candidate.clamp(min=1e-6).abs() * candidate.sign()
            d_v = torch.where(candidate.abs() < 1e-6, torch.zeros_like(d_v), d_v)

            # Accumulate weight gradients
            d_W_x += d_v.T @ x_t  # [D, B] @ [B, D] = [D, D]
            d_W_delta += d_delta_raw.T @ x_t
            d_r_h += (d_v * h_prev_linear).sum(dim=0)
            d_b += d_v.sum(dim=0)
            d_b_delta += d_delta_raw.sum(dim=0)

            # Propagate gradient to previous timestep
            d_log_h = d_log_h_prev

        return {
            'W_x': d_W_x,
            'W_delta': d_W_delta,
            'r_h': d_r_h,
            'b': d_b,
            'b_delta': d_b_delta,
        }


def test_log_space_gradient_flow():
    """Test that log-space backward prevents gradient vanishing."""

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== True Log-Space Backward Test ===")
    print()

    dim = 64
    batch = 4

    for seq_len in [64, 256, 1024]:
        print(f"--- Sequence length {seq_len} ---")

        model = TrueLogSpaceGRU(dim).to(device).float()

        x = torch.randn(seq_len, batch, dim, device=device)

        # Forward
        log_h, sign_h = model(x)

        # Simulate gradient from loss at final timestep
        d_log_h_final = torch.ones(batch, dim, device=device)

        # Log-space backward
        grads = model.backward_log_space(d_log_h_final)

        # Check gradient magnitudes
        print(f"  W_x grad norm: {grads['W_x'].abs().mean().item():.6e}")
        print(f"  r_h grad norm: {grads['r_h'].abs().mean().item():.6e}")
        print(f"  b grad norm: {grads['b'].abs().mean().item():.6e}")
        print()


if __name__ == "__main__":
    test_log_space_gradient_flow()

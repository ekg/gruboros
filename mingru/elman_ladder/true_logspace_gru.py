"""
True Log-Space GRU with Log-Space Backward Pass

Key insight: The backward pass through logaddexp uses softmax weights,
which are bounded [0, 1] and don't cause gradient vanishing!

For z = logaddexp(a, b):
  dL/da = dL/dz * exp(a - z)  = dL/dz * softmax(a,b)[0]
  dL/db = dL/dz * exp(b - z)  = dL/dz * softmax(a,b)[1]

The exp(a-z) term is the softmax weight, always in [0,1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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
    return sign_x * torch.exp(log_x.clamp(max=80))  # Prevent overflow


class LogSpaceGRUFunction(Function):
    """
    Custom autograd function for log-space GRU.

    Forward: Standard computation but store log-space intermediates
    Backward: Compute gradients using log-space softmax weights
    """

    @staticmethod
    def forward(ctx, x, W_x, W_delta, r_h, b, b_delta, log_h0, sign_h0):
        """
        Args:
            x: [T, B, D] input
            W_x: [D, D] candidate weights
            W_delta: [D, D] gate weights
            r_h: [D] diagonal recurrence
            b: [D] candidate bias
            b_delta: [D] gate bias
            log_h0, sign_h0: [B, D] initial hidden state

        Returns:
            log_h_all: [T+1, B, D] log magnitude of hidden states
            sign_h_all: [T+1, B, D] signs of hidden states
        """
        T, B, D = x.shape
        device = x.device
        dtype = x.dtype

        # Storage
        log_h_all = torch.zeros(T + 1, B, D, device=device, dtype=dtype)
        sign_h_all = torch.ones(T + 1, B, D, device=device, dtype=dtype)
        log_h_all[0] = log_h0
        sign_h_all[0] = sign_h0

        # For backward: store softmax weights at each timestep
        # weight1[t] = exp(log_term1 - log_h_new) = contribution of (1-δ)*h_prev
        # weight2[t] = exp(log_term2 - log_h_new) = contribution of δ*candidate
        weight1_all = torch.zeros(T, B, D, device=device, dtype=dtype)
        weight2_all = torch.zeros(T, B, D, device=device, dtype=dtype)

        # Store other intermediates
        delta_all = torch.zeros(T, B, D, device=device, dtype=dtype)
        v_all = torch.zeros(T, B, D, device=device, dtype=dtype)
        h_prev_linear_all = torch.zeros(T, B, D, device=device, dtype=dtype)

        for t in range(T):
            log_h_prev = log_h_all[t]
            sign_h_prev = sign_h_all[t]
            x_t = x[t]

            # Delta gate
            delta_raw = x_t @ W_delta.T + b_delta
            delta = torch.sigmoid(delta_raw)
            log_delta = log_sigmoid(delta_raw)
            log_one_minus_delta = log_one_minus_sigmoid(delta_raw)

            # Candidate
            h_prev_linear = from_log_space(log_h_prev, sign_h_prev)
            v = x_t @ W_x.T + r_h * h_prev_linear + b
            candidate = torch.tanh(v)
            log_candidate, sign_candidate = to_log_space(candidate)

            # Log-space update
            log_term1 = log_one_minus_delta + log_h_prev
            log_term2 = log_delta + log_candidate

            # Compute logaddexp result and softmax weights
            log_max = torch.maximum(log_term1, log_term2)
            log_h_new = log_max + torch.log(
                torch.exp(log_term1 - log_max) + torch.exp(log_term2 - log_max)
            )

            # Softmax weights for backward
            weight1 = torch.exp(log_term1 - log_h_new).clamp(0, 1)
            weight2 = torch.exp(log_term2 - log_h_new).clamp(0, 1)

            # Determine sign of result
            # If term1 dominates and has same sign as term2, keep that sign
            # If opposite signs, take sign of dominant term
            term1_bigger = log_term1 > log_term2
            same_sign = sign_h_prev * sign_candidate > 0
            sign_h_new = torch.where(
                same_sign,
                sign_h_prev,  # Both same sign
                torch.where(term1_bigger, sign_h_prev, sign_candidate)  # Dominant term's sign
            )

            # Store
            log_h_all[t + 1] = log_h_new
            sign_h_all[t + 1] = sign_h_new
            weight1_all[t] = weight1
            weight2_all[t] = weight2
            delta_all[t] = delta
            v_all[t] = v
            h_prev_linear_all[t] = h_prev_linear

        ctx.save_for_backward(
            x, W_x, W_delta, r_h, b, b_delta,
            log_h_all, sign_h_all,
            weight1_all, weight2_all,
            delta_all, v_all, h_prev_linear_all
        )

        return log_h_all, sign_h_all

    @staticmethod
    def backward(ctx, grad_log_h_all, grad_sign_h_all):
        """
        Backward pass using log-space softmax weights.

        Key: gradient through logaddexp uses softmax weights which are bounded!
        """
        (x, W_x, W_delta, r_h, b, b_delta,
         log_h_all, sign_h_all,
         weight1_all, weight2_all,
         delta_all, v_all, h_prev_linear_all) = ctx.saved_tensors

        T, B, D = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_W_x = torch.zeros_like(W_x)
        grad_W_delta = torch.zeros_like(W_delta)
        grad_r_h = torch.zeros_like(r_h)
        grad_b = torch.zeros_like(b)
        grad_b_delta = torch.zeros_like(b_delta)

        # Gradient w.r.t. log_h at each timestep
        # Start from the final gradient
        grad_log_h = grad_log_h_all[-1].clone()

        for t in reversed(range(T)):
            # Add incoming gradient from output at this timestep
            grad_log_h = grad_log_h + grad_log_h_all[t + 1]

            weight1 = weight1_all[t]  # softmax weight for (1-δ)*h_prev
            weight2 = weight2_all[t]  # softmax weight for δ*candidate
            delta = delta_all[t]
            v = v_all[t]
            h_prev_linear = h_prev_linear_all[t]
            x_t = x[t]
            sign_h_prev = sign_h_all[t]
            log_h_prev = log_h_all[t]

            candidate = torch.tanh(v)

            # Backward through logaddexp
            # z = logaddexp(a, b) => dz/da = weight1, dz/db = weight2
            grad_log_term1 = grad_log_h * weight1
            grad_log_term2 = grad_log_h * weight2

            # term1 = log_one_minus_delta + log_h_prev
            # d(log_one_minus_delta) = grad_log_term1
            # d(log_h_prev) = grad_log_term1  <-- KEY: gradient flows through!
            grad_log_h_prev = grad_log_term1

            # term2 = log_delta + log_candidate
            grad_log_delta = grad_log_term2
            grad_log_candidate = grad_log_term2

            # Backward through log_sigmoid and log_one_minus_sigmoid
            # log_delta = -softplus(-delta_raw) => d(delta_raw) = delta * grad_log_delta
            # log_one_minus_delta = -softplus(delta_raw) => d(delta_raw) = -(1-delta) * grad_log_term1
            grad_delta_raw = delta * grad_log_delta - (1 - delta) * grad_log_term1

            # Backward through tanh and log
            # candidate = tanh(v), log_candidate = log|candidate|
            # grad_v = grad_log_candidate * (1 - tanh^2(v)) / |candidate|
            dtanh = 1 - candidate ** 2
            grad_v = grad_log_candidate * dtanh / (candidate.abs().clamp(min=1e-6))
            # Handle sign
            grad_v = grad_v * candidate.sign()
            grad_v = torch.where(candidate.abs() < 1e-6, torch.zeros_like(grad_v), grad_v)

            # Backward through v = x @ W_x.T + r_h * h_prev_linear + b
            grad_x[t] = grad_v @ W_x + grad_delta_raw @ W_delta
            grad_W_x += grad_v.T @ x_t
            grad_W_delta += grad_delta_raw.T @ x_t
            grad_r_h += (grad_v * h_prev_linear).sum(dim=0)
            grad_b += grad_v.sum(dim=0)
            grad_b_delta += grad_delta_raw.sum(dim=0)

            # Gradient through h_prev_linear = sign_h_prev * exp(log_h_prev)
            # d(log_h_prev) += grad_v * r_h * d(h_prev_linear)/d(log_h_prev)
            # d(h_prev_linear)/d(log_h_prev) = h_prev_linear
            grad_log_h_prev += grad_v * r_h * h_prev_linear

            # Propagate to previous timestep
            grad_log_h = grad_log_h_prev

        return grad_x, grad_W_x, grad_W_delta, grad_r_h, grad_b, grad_b_delta, None, None


class TrueLogSpaceGRU(nn.Module):
    """True log-space GRU with log-space backward pass."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_x = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.W_delta = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.r_h = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), -2.0))

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            log_h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, D, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = to_log_space(h0)

        log_h_all, sign_h_all = LogSpaceGRUFunction.apply(
            x, self.W_x, self.W_delta, self.r_h, self.b, self.b_delta,
            log_h0, sign_h0
        )

        # Convert final hidden state to linear
        h_final = from_log_space(log_h_all[-1], sign_h_all[-1])

        return h_final, (log_h_all, sign_h_all)


def test_gradient_flow():
    """Test that the true log-space backward prevents gradient vanishing."""

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== True Log-Space GRU Gradient Flow Test ===")
    print()

    dim = 64
    batch = 4

    for seq_len in [64, 256, 1024, 4096]:
        print(f"--- Sequence length {seq_len} ---")

        model = TrueLogSpaceGRU(dim).to(device).float()
        x = torch.randn(seq_len, batch, dim, device=device, requires_grad=True)

        h_final, _ = model(x)
        loss = h_final.sum()
        loss.backward()

        # Check gradient at first and last timestep
        x_grad_first = x.grad[0].abs().mean().item()
        x_grad_last = x.grad[-1].abs().mean().item()
        ratio = x_grad_last / x_grad_first if x_grad_first > 0 else float('inf')

        print(f"  x[0] grad: {x_grad_first:.6e}")
        print(f"  x[-1] grad: {x_grad_last:.6e}")
        print(f"  ratio (last/first): {ratio:.2f}x")
        print(f"  W_x grad: {model.W_x.grad.abs().mean().item():.6e}")
        print()


if __name__ == "__main__":
    test_gradient_flow()

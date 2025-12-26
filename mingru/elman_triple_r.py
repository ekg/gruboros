"""
ElmanTripleR: Three separate R matrices for different signal pathways

Architecture:
  candidate = tanh(R_h @ h + R_x @ x + b)
  delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)
  h_new = (1 - delta) * h + delta * candidate

This separates:
- R_h: hidden-to-hidden temporal patterns
- R_x: input transformation (richer than simple Wx)
- R_delta: context-aware forget decisions (h-dependent delta!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import haste_pytorch_lib as haste
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch_lib not available, ElmanTripleR will use fallback")


class ElmanTripleRFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, R_h, R_x, R_delta, W_delta, b, b_delta):
        h, v, delta_cache = haste.elman_triple_r_forward(
            training, x, h0, R_h, R_x, R_delta, W_delta, b, b_delta)

        if training:
            ctx.save_for_backward(x, R_h, R_x, R_delta, W_delta, h, v, delta_cache)

        return h

    @staticmethod
    def backward(ctx, grad_h):
        x, R_h, R_x, R_delta, W_delta, h, v, delta_cache = ctx.saved_tensors

        dx, dh0, dR_h, dR_x, dR_delta, dW_delta, db, db_delta = haste.elman_triple_r_backward(
            x, R_h, R_x, R_delta, W_delta, h, v, delta_cache, grad_h.contiguous())

        return None, dx, dh0, dR_h, dR_x, dR_delta, dW_delta, db, db_delta


class ElmanTripleR(nn.Module):
    """
    ElmanTripleR RNN with three separate R matrices.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        delta_init: Initial value for delta bias (controls initial retention)
    """

    def __init__(self, input_size, hidden_size, delta_init=-2.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Three R matrices
        self.R_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.R_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R_delta = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Delta projection
        self.W_delta = nn.Parameter(torch.empty(hidden_size, input_size))

        # Biases
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.b_delta = nn.Parameter(torch.full((hidden_size,), delta_init))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for R matrices
        nn.init.xavier_uniform_(self.R_h)
        nn.init.xavier_uniform_(self.R_x)
        nn.init.xavier_uniform_(self.R_delta)
        nn.init.xavier_uniform_(self.W_delta)

    def forward(self, x, h0=None):
        """
        Args:
            x: Input tensor [T, B, input_size]
            h0: Initial hidden state [B, hidden_size], or None for zeros

        Returns:
            h: Hidden states [T+1, B, hidden_size] (includes initial state)
        """
        T, B, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)

        if HASTE_AVAILABLE and x.is_cuda:
            h = ElmanTripleRFunction.apply(
                self.training, x.contiguous(), h0.contiguous(),
                self.R_h, self.R_x, self.R_delta, self.W_delta,
                self.b, self.b_delta)
        else:
            # Fallback PyTorch implementation
            h = [h0]
            for t in range(T):
                h_prev = h[-1]
                x_t = x[t]

                # Candidate
                raw = F.linear(h_prev, self.R_h) + F.linear(x_t, self.R_x) + self.b
                candidate = torch.tanh(raw)

                # Delta (h-dependent!)
                delta_raw = F.linear(h_prev, self.R_delta) + F.linear(x_t, self.W_delta) + self.b_delta
                delta = torch.sigmoid(delta_raw)

                # Leaky integration
                h_new = (1 - delta) * h_prev + delta * candidate
                h.append(h_new)

            h = torch.stack(h, dim=0)

        return h

    def extra_repr(self):
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}'

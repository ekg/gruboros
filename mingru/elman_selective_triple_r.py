"""
ElmanSelectiveTripleR: Triple R with input-dependent B gate (like Mamba2's selection)

Architecture:
  B_gate = sigmoid(W_B @ x + b_B)                      -- input-dependent write gate
  candidate = tanh(R_h @ h + B_gate * (R_x @ x) + b)   -- B_gate modulates input
  delta = sigmoid(R_delta @ h + W_delta @ x + b_delta) -- context-aware delta
  h_new = (1 - delta) * h + delta * candidate

The B gate provides input selectivity similar to Mamba2's B projection:
- When B_gate ≈ 0: input is ignored, only hidden state propagates
- When B_gate ≈ 1: input is fully incorporated
- Intermediate values: selective attention to input features
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
    print("Warning: haste_pytorch_lib not available, ElmanSelectiveTripleR will use fallback")


class ElmanSelectiveTripleRFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, R_h, R_x, R_delta, W_delta, W_B, b, b_delta, b_B):
        h, v, delta_cache, B_gate_cache = haste.elman_selective_triple_r_forward(
            training, x, h0, R_h, R_x, R_delta, W_delta, W_B, b, b_delta, b_B)

        if training:
            ctx.save_for_backward(x, R_h, R_x, R_delta, W_delta, W_B, h, v, delta_cache, B_gate_cache)

        return h

    @staticmethod
    def backward(ctx, grad_h):
        x, R_h, R_x, R_delta, W_delta, W_B, h, v, delta_cache, B_gate_cache = ctx.saved_tensors

        dx, dh0, dR_h, dR_x, dR_delta, dW_delta, dW_B, db, db_delta, db_B = haste.elman_selective_triple_r_backward(
            x, R_h, R_x, R_delta, W_delta, W_B, h, v, delta_cache, B_gate_cache, grad_h.contiguous())

        return None, dx, dh0, dR_h, dR_x, dR_delta, dW_delta, dW_B, db, db_delta, db_B


class ElmanSelectiveTripleR(nn.Module):
    """
    ElmanSelectiveTripleR RNN with input-dependent B gate.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        delta_init: Initial value for delta bias (controls initial retention)
        b_gate_init: Initial value for B gate bias (0 = ~50% selectivity)
    """

    def __init__(self, input_size, hidden_size, delta_init=-2.0, b_gate_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Three R matrices (same as Triple R)
        self.R_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.R_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R_delta = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Delta projection
        self.W_delta = nn.Parameter(torch.empty(hidden_size, input_size))

        # NEW: B gate projection (input-dependent selectivity)
        self.W_B = nn.Parameter(torch.empty(hidden_size, input_size))

        # Biases
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.b_delta = nn.Parameter(torch.full((hidden_size,), delta_init))
        self.b_B = nn.Parameter(torch.full((hidden_size,), b_gate_init))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all matrices
        nn.init.xavier_uniform_(self.R_h)
        nn.init.xavier_uniform_(self.R_x)
        nn.init.xavier_uniform_(self.R_delta)
        nn.init.xavier_uniform_(self.W_delta)
        nn.init.xavier_uniform_(self.W_B)

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
            h = ElmanSelectiveTripleRFunction.apply(
                self.training, x.contiguous(), h0.contiguous(),
                self.R_h, self.R_x, self.R_delta, self.W_delta, self.W_B,
                self.b, self.b_delta, self.b_B)
        else:
            # Fallback PyTorch implementation
            h = [h0]
            for t in range(T):
                h_prev = h[-1]
                x_t = x[t]

                # B gate (input-dependent selectivity)
                B_gate = torch.sigmoid(F.linear(x_t, self.W_B) + self.b_B)

                # Candidate (B_gate modulates input contribution)
                R_h_out = F.linear(h_prev, self.R_h)
                R_x_out = F.linear(x_t, self.R_x)
                raw = R_h_out + B_gate * R_x_out + self.b
                candidate = torch.tanh(raw)

                # Delta (context-aware forget)
                delta_raw = F.linear(h_prev, self.R_delta) + F.linear(x_t, self.W_delta) + self.b_delta
                delta = torch.sigmoid(delta_raw)

                # Leaky integration
                h_new = (1 - delta) * h_prev + delta * candidate
                h.append(h_new)

            h = torch.stack(h, dim=0)

        return h

    def extra_repr(self):
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}'

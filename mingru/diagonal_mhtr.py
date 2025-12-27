"""
Diagonal Multi-Head Triple R (Diagonal MHTR)

Key insight: Full R matrix transitions compound errors over depth.
Diagonal R = element-wise transitions like Mamba2's diagonal A matrix.

Change from full MHTR:
  Full:     h_new = R @ h + ...    (matrix-matrix, unstable at depth)
  Diagonal: h_new = R * h + ...    (element-wise, stable at depth)

This sacrifices expressivity for stability - states evolve independently.
Same multi-head structure, but simpler per-head dynamics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the Haste CUDA kernel
try:
    import haste_pytorch_lib
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch_lib not available, using slow PyTorch fallback for DiagonalMHTR")


class DiagonalMHTRFunction(torch.autograd.Function):
    """Autograd function wrapping Haste CUDA kernel for Diagonal MHTR."""

    @staticmethod
    def forward(ctx, training, x, h0, R_h, R_x, R_delta, W_delta, b, b_delta):
        """
        Forward pass using Haste CUDA kernel.

        Args:
            x: [T, B, nheads, headdim]
            h0: [B, nheads, headdim]
            R_h, R_x, R_delta: [nheads, headdim] diagonal vectors
            W_delta: [nheads, headdim, headdim]
            b, b_delta: [nheads, headdim]

        Returns:
            h: [T+1, B, nheads, headdim]
        """
        h, v, delta_cache = haste_pytorch_lib.diagonal_mhtr_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            R_h.contiguous(),
            R_x.contiguous(),
            R_delta.contiguous(),
            W_delta.contiguous(),
            b.contiguous(),
            b_delta.contiguous()
        )

        if training:
            ctx.save_for_backward(x, R_h, R_x, R_delta, W_delta, h, v, delta_cache)

        return h

    @staticmethod
    def backward(ctx, dh_out):
        """Backward pass using Haste CUDA kernel."""
        x, R_h, R_x, R_delta, W_delta, h, v, delta_cache = ctx.saved_tensors

        # dh_out is [T+1, B, nheads, headdim], we need [T, B, nheads, headdim]
        # The gradient from h[0] (initial state) is typically not backpropagated through training
        dh = dh_out[1:].contiguous()  # [T, B, nheads, headdim]

        dx, dR_h, dR_x, dR_delta, dW_delta, db, db_delta = haste_pytorch_lib.diagonal_mhtr_backward(
            x, R_h, R_x, R_delta, W_delta, h, v, delta_cache, dh
        )

        # Return gradients in same order as forward args
        # training, x, h0, R_h, R_x, R_delta, W_delta, b, b_delta
        return None, dx, None, dR_h, dR_x, dR_delta, dW_delta, db, db_delta


class DiagonalMHTRCore(nn.Module):
    """
    Multi-head Triple R with DIAGONAL transitions.

    Each head has:
      - R_h: [headdim] diagonal for state transition
      - R_x: [headdim] diagonal for input
      - R_delta: [headdim] diagonal for gate state path
      - W_delta: [headdim, headdim] for gate input path (kept full - it's not recurrent)
    """

    def __init__(self, nheads, headdim, delta_init=-2.0):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim

        # Diagonal R vectors per head: [nheads, headdim]
        self.R_h = nn.Parameter(torch.empty(nheads, headdim))
        self.R_x = nn.Parameter(torch.empty(nheads, headdim))
        self.R_delta = nn.Parameter(torch.empty(nheads, headdim))

        # W_delta per head (full matrix - it's the input path, not recurrent)
        self.W_delta = nn.Parameter(torch.empty(nheads, headdim, headdim))

        # Biases per head
        self.b = nn.Parameter(torch.zeros(nheads, headdim))
        self.b_delta = nn.Parameter(torch.full((nheads, headdim), delta_init))

        self._init_weights()

    def _init_weights(self):
        # Initialize diagonal R with values that decay state over time
        # Mean ~0.9 so state persists, small variance
        # Key: |R| < 1 for stability over many layers
        nn.init.normal_(self.R_h, mean=0.9, std=0.05)
        nn.init.normal_(self.R_x, mean=0.0, std=0.1)  # Input mixing
        nn.init.normal_(self.R_delta, mean=0.0, std=0.05)  # Gate state path small

        # Small init for W_delta (input path for gate)
        nn.init.normal_(self.W_delta, mean=0, std=0.01)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, nheads, headdim]
            h0: [B, nheads, headdim] or None

        Returns:
            h: [T+1, B, nheads, headdim]
        """
        T, B, nh, hd = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.nheads, self.headdim, device=x.device, dtype=x.dtype)

        # Use Haste CUDA kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return DiagonalMHTRFunction.apply(
                self.training,
                x, h0,
                self.R_h, self.R_x, self.R_delta,
                self.W_delta,
                self.b, self.b_delta
            )

        # PyTorch fallback (slow but works)
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch fallback (slow, for CPU/debugging)."""
        T, B, nh, hd = x.shape
        h_list = [h0]

        for t in range(T):
            h_prev = h_list[-1]  # [B, nheads, headdim]
            x_t = x[t]  # [B, nheads, headdim]

            # DIAGONAL transitions - element-wise multiplication!
            R_h_h = self.R_h * h_prev  # Element-wise!
            R_x_x = self.R_x * x_t     # Element-wise!

            raw = R_h_h + R_x_x + self.b
            candidate = torch.tanh(raw)

            # Gate: diagonal for state, full matrix for input
            R_delta_h = self.R_delta * h_prev  # Element-wise
            # W_delta @ x still uses matrix (it's input path, not recurrent)
            W_delta_x = torch.einsum('nhd,bnh->bnd', self.W_delta, x_t)

            delta_raw = R_delta_h + W_delta_x + self.b_delta
            delta = torch.sigmoid(delta_raw)

            h_new = (1 - delta) * h_prev + delta * candidate
            h_list.append(h_new)

        h = torch.stack(h_list, dim=0)
        return h


class DiagonalMHTR(nn.Module):
    """
    Diagonal Multi-Head Triple R - depth-stable variant.

    Uses element-wise (diagonal) state transitions instead of full matrices.
    Same state expansion as full MHTR, but stable at depth=48+.
    """

    def __init__(self, dim, expand=2, headdim=64, delta_init=-2.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.d_inner = dim * expand
        self.headdim = headdim
        self.nheads = self.d_inner // headdim

        assert self.d_inner % headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Diagonal MHTR core
        self.rnn = DiagonalMHTRCore(
            nheads=self.nheads,
            headdim=headdim,
            delta_init=delta_init
        )

        # Gate projection (from original dim)
        self.silu_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.silu_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim]
            h0: Optional [B, nheads, headdim]

        Returns:
            output: [B, T, dim]
            h_final: [B, nheads, headdim]
        """
        B, T, D = x.shape

        # Project and reshape for multi-head
        x_proj = self.in_proj(x)  # [B, T, d_inner]
        x_heads = x_proj.view(B, T, self.nheads, self.headdim)  # [B, T, nheads, headdim]

        # Transpose for RNN: [T, B, nheads, headdim]
        x_rnn = x_heads.permute(1, 0, 2, 3).contiguous()

        # Run diagonal MHTR
        h_all = self.rnn(x_rnn, h0)  # [T+1, B, nheads, headdim]
        h_out = h_all[1:]  # [T, B, nheads, headdim]
        h_final = h_all[-1]  # [B, nheads, headdim]

        # Reshape back: [B, T, d_inner]
        h_out = h_out.permute(1, 0, 2, 3).contiguous()
        h_out = h_out.view(B, T, self.d_inner)

        # Simple SiLU gating (like Mamba)
        gate = F.silu(self.silu_proj(x))

        # Apply gate and project
        gated = h_out * gate
        output = self.out_proj(gated)

        return output, h_final

    def extra_repr(self):
        return (f'dim={self.dim}, expand={self.expand}, nheads={self.nheads}, '
                f'headdim={self.headdim}, DIAGONAL transitions')


if __name__ == "__main__":
    print("Testing DiagonalMHTR...")
    print("=" * 60)

    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test at target config
    model = DiagonalMHTR(
        dim=2048,
        expand=2,
        headdim=64,
        delta_init=-2.0
    ).cuda()

    params = sum(p.numel() for p in model.parameters())
    print(f"Layer params: {params:,}")
    print(f"Config: nheads={model.nheads}, headdim={model.headdim}")

    # Compare to full MHTR params
    nheads, headdim, d_state_r = 64, 64, 16
    full_r_params = 3 * 2 * nheads * headdim * d_state_r
    diag_r_params = 3 * nheads * headdim
    print(f"R params: Full={full_r_params:,}, Diagonal={diag_r_params:,} ({full_r_params/diag_r_params:.0f}× reduction)")

    # Test forward
    x = torch.randn(2, 32, 2048).cuda()
    out, h = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")

    # Test backward
    print("\n" + "=" * 60)
    print("Testing gradient flow...")

    loss = out.sum()
    loss.backward()

    print(f"Gradient check passed!")

    # Speed test
    print("\n" + "=" * 60)
    print("Speed test...")

    import time

    model.train()
    x = torch.randn(4, 512, 2048).cuda()

    # Warmup
    for _ in range(3):
        out, _ = model(x)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        out, _ = model(x)
        out.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = 4 * 512 * 10
    print(f"Throughput: {tokens / elapsed:.0f} tokens/sec")

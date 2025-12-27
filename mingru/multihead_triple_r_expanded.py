"""
Multi-Head Triple R with 32× State Expansion (Mamba2-style)

Key insight from Mamba2: expand hidden state to 32× model dim, but use simple transitions.
We keep the Triple R rich transitions but apply them per-head on expanded state.

Architecture:
  - expand=2: d_model (2048) → d_inner (4096)
  - nheads=64: Split into 64 independent heads (headdim=64)
  - d_state=16: Each head has state of size headdim × d_state = 1024
  - Total state: 64 × 1024 = 65,536 (32× d_model, same as Mamba2!)

Per-head recurrence (low-rank R matrices):
  candidate = tanh(R_h @ h + R_x @ x + b)
  delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)
  h_new = (1 - delta) * h + delta * candidate

Uses Haste kernel by materializing R = R_down @ R_up per head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import haste_pytorch_lib as haste
    HASTE_AVAILABLE = True
except ImportError:
    HASTE_AVAILABLE = False
    print("Warning: haste_pytorch_lib not available, MultiHeadTripleRExpanded will use PyTorch fallback")


class MultiHeadTripleRExpandedFunction(Function):
    """Autograd function for multi-head Triple R with fused Haste kernel."""

    @staticmethod
    def forward(ctx, training, x, h0,
                R_h_down, R_h_up, R_x_down, R_x_up,
                R_delta_down, R_delta_up, W_delta, b, b_delta,
                nheads, headdim, d_state_r):
        """
        x: [T, B, nheads, headdim] - input per head
        h0: [B, nheads, headdim] - initial state per head
        R matrices: [nheads, headdim, d_state_r] or [nheads, d_state_r, headdim]
        """
        T, B, nh, hd = x.shape

        # Materialize full R matrices per head
        # R_h: [nheads, headdim, headdim]
        R_h = torch.bmm(R_h_down, R_h_up)  # [nheads, headdim, headdim]
        R_x = torch.bmm(R_x_down, R_x_up)  # [nheads, headdim, headdim]
        R_delta = torch.bmm(R_delta_down, R_delta_up)  # [nheads, headdim, headdim]

        # Call fused multi-head kernel (all heads in one call!)
        h_out, v, delta_cache = haste.multihead_triple_r_forward(
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
            ctx.save_for_backward(
                x, R_h_down, R_h_up, R_x_down, R_x_up,
                R_delta_down, R_delta_up, W_delta,
                R_h, R_x, R_delta, h_out, v, delta_cache
            )
            ctx.nheads = nheads
            ctx.headdim = headdim

        return h_out

    @staticmethod
    def backward(ctx, grad_h):
        (x, R_h_down, R_h_up, R_x_down, R_x_up,
         R_delta_down, R_delta_up, W_delta,
         R_h, R_x, R_delta, h_out, v, delta_cache) = ctx.saved_tensors
        nheads = ctx.nheads
        headdim = ctx.headdim

        # Call fused multi-head backward kernel
        dx, dh0, dR_h, dR_x, dR_delta, dW_delta, db, db_delta = haste.multihead_triple_r_backward(
            x.contiguous(),
            R_h.contiguous(),
            R_x.contiguous(),
            R_delta.contiguous(),
            W_delta.contiguous(),
            h_out.contiguous(),
            v.contiguous(),
            delta_cache.contiguous(),
            grad_h.contiguous()
        )

        # Chain rule for low-rank: d(R_down @ R_up) = dR
        dR_h_down = torch.bmm(dR_h, R_h_up.transpose(1, 2))
        dR_h_up = torch.bmm(R_h_down.transpose(1, 2), dR_h)

        dR_x_down = torch.bmm(dR_x, R_x_up.transpose(1, 2))
        dR_x_up = torch.bmm(R_x_down.transpose(1, 2), dR_x)

        dR_delta_down = torch.bmm(dR_delta, R_delta_up.transpose(1, 2))
        dR_delta_up = torch.bmm(R_delta_down.transpose(1, 2), dR_delta)

        return (None, dx, dh0,
                dR_h_down, dR_h_up, dR_x_down, dR_x_up,
                dR_delta_down, dR_delta_up, dW_delta, db, db_delta,
                None, None, None)


class MultiHeadTripleRCore(nn.Module):
    """
    Multi-head Triple R RNN core with low-rank R matrices per head.
    """

    def __init__(self, nheads, headdim, d_state_r=16, delta_init=-2.0):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        self.d_state_r = d_state_r

        # Low-rank R matrices per head: [nheads, headdim, d_state_r]
        self.R_h_down = nn.Parameter(torch.empty(nheads, headdim, d_state_r))
        self.R_h_up = nn.Parameter(torch.empty(nheads, d_state_r, headdim))

        self.R_x_down = nn.Parameter(torch.empty(nheads, headdim, d_state_r))
        self.R_x_up = nn.Parameter(torch.empty(nheads, d_state_r, headdim))

        self.R_delta_down = nn.Parameter(torch.empty(nheads, headdim, d_state_r))
        self.R_delta_up = nn.Parameter(torch.empty(nheads, d_state_r, headdim))

        # W_delta per head (full, not low-rank - it's input to delta)
        self.W_delta = nn.Parameter(torch.empty(nheads, headdim, headdim))

        # Biases per head
        self.b = nn.Parameter(torch.zeros(nheads, headdim))
        self.b_delta = nn.Parameter(torch.full((nheads, headdim), delta_init))

        self._init_weights()

    def _init_weights(self):
        # For R = R_down @ R_up with spectral norm ~1:
        # Var[R_ij] = d_state_r * std^4 should ≈ 1/headdim
        # std = (1 / (headdim * d_state_r))^0.25
        std = (1.0 / (self.headdim * self.d_state_r)) ** 0.25  # ~0.177 for headdim=64, d_state_r=16

        nn.init.normal_(self.R_h_down, mean=0, std=std)
        nn.init.normal_(self.R_h_up, mean=0, std=std)
        nn.init.normal_(self.R_x_down, mean=0, std=std)
        nn.init.normal_(self.R_x_up, mean=0, std=std)
        nn.init.normal_(self.R_delta_down, mean=0, std=std)
        nn.init.normal_(self.R_delta_up, mean=0, std=std)

        # Small init for W_delta (input path, not recurrent)
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

        if HASTE_AVAILABLE and x.is_cuda:
            h = MultiHeadTripleRExpandedFunction.apply(
                self.training, x.contiguous(), h0.contiguous(),
                self.R_h_down, self.R_h_up,
                self.R_x_down, self.R_x_up,
                self.R_delta_down, self.R_delta_up,
                self.W_delta, self.b, self.b_delta,
                self.nheads, self.headdim, self.d_state_r
            )
        else:
            # PyTorch fallback
            h_list = [h0]

            # Materialize R matrices
            R_h = torch.bmm(self.R_h_down, self.R_h_up)
            R_x = torch.bmm(self.R_x_down, self.R_x_up)
            R_delta = torch.bmm(self.R_delta_down, self.R_delta_up)

            for t in range(T):
                h_prev = h_list[-1]  # [B, nheads, headdim]
                x_t = x[t]  # [B, nheads, headdim]

                # Per-head computation using einsum for efficiency
                # R_h @ h: [nheads, headdim, headdim] @ [B, nheads, headdim]
                R_h_h = torch.einsum('nhd,bnh->bnd', R_h, h_prev)  # [B, nheads, headdim]
                R_x_x = torch.einsum('nhd,bnh->bnd', R_x, x_t)

                raw = R_h_h + R_x_x + self.b
                candidate = torch.tanh(raw)

                R_delta_h = torch.einsum('nhd,bnh->bnd', R_delta, h_prev)
                W_delta_x = torch.einsum('nhd,bnh->bnd', self.W_delta, x_t)

                delta_raw = R_delta_h + W_delta_x + self.b_delta
                delta = torch.sigmoid(delta_raw)

                h_new = (1 - delta) * h_prev + delta * candidate
                h_list.append(h_new)

            h = torch.stack(h_list, dim=0)

        return h


class MultiHeadTripleRExpanded(nn.Module):
    """
    Multi-Head Triple R with 32× state expansion.
    Drop-in replacement matching Mamba2's state capacity.

    Config for 0.915B at depth=21:
      d_model=2048, expand=2, nheads=64, headdim=64, d_state_r=16
    """

    def __init__(self, dim, expand=2, headdim=64, d_state_r=16,
                 delta_init=-2.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.d_inner = dim * expand
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_state_r = d_state_r

        assert self.d_inner % headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Multi-head Triple R core
        self.rnn = MultiHeadTripleRCore(
            nheads=self.nheads,
            headdim=headdim,
            d_state_r=d_state_r,
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
        # Xavier for out_proj (needs reasonable signal for gradients)
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

        # Run multi-head RNN
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
        state_size = self.nheads * self.headdim * self.d_state_r
        return (f'dim={self.dim}, expand={self.expand}, nheads={self.nheads}, '
                f'headdim={self.headdim}, d_state_r={self.d_state_r}, '
                f'total_state={state_size} ({state_size/self.dim:.0f}× dim)')


if __name__ == "__main__":
    print("Testing MultiHeadTripleRExpanded...")
    print("=" * 60)

    # Test at target config
    model = MultiHeadTripleRExpanded(
        dim=2048,
        expand=2,
        headdim=64,
        d_state_r=16,
        n_groups=32,
        delta_init=-1.8
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Layer params: {params:,}")
    print(f"Config: nheads={model.nheads}, headdim={model.headdim}")
    print(f"State expansion: {model.nheads * model.headdim * model.d_state_r / model.dim:.0f}× dim")

    # Test forward
    x = torch.randn(2, 32, 2048)
    out, h = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")

    # Full model param count
    print("\n" + "=" * 60)
    print("Full model param estimates:")

    vocab_size = 50281
    dim = 2048
    embed_params = 2 * vocab_size * dim

    for depth in [20, 21, 22]:
        total = embed_params + depth * params
        print(f"depth={depth}: {total:,} ({total/1e9:.3f}B)")

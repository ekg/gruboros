"""
Haste GRU + Multiplicative Gating (SiLU) - EXACT match to CuDNNGRU_MultLM.

This file mirrors cudnn_gru_mult.py EXACTLY, just swapping the GRU backend:
- CuDNNGRU_MultLM uses: nn.GRU + separate gate computation
- HasteGRU_MultLM uses: GRU_SiLU (fused GRU + SiLU gate in CUDA kernel)

The haste GRU_SiLU kernel computes:
  h_gru = GRU(x, h_prev)                    # Standard GRU recurrence
  gate = silu(Wg_x @ x + Wg_h @ h_gru + bg) # SiLU selectivity gate
  output = h_gru * gate                     # Multiplicative gating

This is mathematically IDENTICAL to CuDNNGRU_Mult with gate_activation='silu'.

IMPORTANT: Model structure (weight tying, LayerNorm, residuals) is IDENTICAL
to CuDNNGRU_MultLM. Only the GRU kernel backend differs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import math

try:
    from haste_pytorch import GRU_SiLU
    HASTE_GRU_SILU_AVAILABLE = True
except ImportError:
    HASTE_GRU_SILU_AVAILABLE = False
    print("Warning: haste_pytorch GRU_SiLU not available.")


class HasteGRU_Mult(nn.Module):
    """
    Haste GRU with fused SiLU multiplicative gating.

    EXACT mathematical equivalence to CuDNNGRU_Mult with gate_activation='silu'.

    CuDNN version computes:
      h_seq = GRU(x, h0)
      gate = silu(W_x @ x + W_h @ h_seq + b)
      out = h_seq * gate

    Haste GRU_SiLU fuses this into a single CUDA kernel with BF16 support.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,  # Must be 1.0 for haste (fused)
        use_input_gate: bool = True,  # Always True for haste (fused)
        use_hidden_gate: bool = True,  # Always True for haste (fused)
        use_glu_gate: bool = False,  # Not supported with haste
        use_swiglu: bool = False,  # Not supported with haste
        gate_activation: str = 'silu',  # Always silu for haste (fused)
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs
    ):
        super().__init__()

        if not HASTE_GRU_SILU_AVAILABLE:
            raise RuntimeError("haste_pytorch GRU_SiLU required")

        # Validate: haste GRU_SiLU has fixed architecture
        if use_glu_gate or use_swiglu:
            raise ValueError("GLU/SwiGLU not supported with haste GRU_SiLU - use CuDNNGRU_Mult")
        if gate_activation != 'silu':
            raise ValueError("Haste GRU_SiLU only supports silu gate activation")
        if gate_expansion != 1.0:
            raise ValueError("Haste GRU_SiLU only supports gate_expansion=1.0")

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # === Haste GRU_SiLU (fused kernel) ===
        # This fuses: GRU + gate computation + multiplicative gating
        self.gru_silu = GRU_SiLU(
            input_size=dim,
            hidden_size=self.dim_inner,
            batch_first=False,  # Haste uses time-first
        )

        # === Output projection ===
        # Match CuDNNGRU_Mult: Identity when expansion_factor=1.0
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to match CuDNNGRU_Mult exactly."""
        # CuDNN version does: nn.init.normal_(self.gate_x.weight, std=0.02)
        # Haste GRU_SiLU has: gate_kernel_x, gate_kernel_h, gate_bias
        with torch.no_grad():
            nn.init.normal_(self.gru_silu.gate_kernel_x, std=0.02)
            nn.init.normal_(self.gru_silu.gate_kernel_h, std=0.02)
            nn.init.zeros_(self.gru_silu.gate_bias)

        # Output projection: small init for residual learning
        if not isinstance(self.to_out, nn.Identity):
            nn.init.normal_(self.to_out.weight, std=0.02 / math.sqrt(2))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass - EXACT interface match to CuDNNGRU_Mult.

        Args:
            x: [B, T, D] input (batch-first)
            prev_hidden: [B, dim_inner] previous hidden state
            return_next_prev_hidden: if True, return (output, next_hidden)

        Returns:
            output: [B, T, D]
            or (output, next_hidden) if return_next_prev_hidden
        """
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # === Haste GRU_SiLU ===
        # Convert to time-first for haste: [T, B, D]
        x_t = x.transpose(0, 1).contiguous()

        # Prepare initial hidden state: haste expects [1, B, H]
        if prev_hidden is not None:
            if prev_hidden.dim() == 2:
                h0 = prev_hidden.unsqueeze(0)
            else:
                h0 = prev_hidden
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)

        # Run fused GRU + SiLU gate
        # Output already has gate applied: h_seq * silu(gate)
        h_seq, h_final = self.gru_silu(x_t, state=h0)

        # Convert back to batch-first: [B, T, H]
        h_seq = h_seq.transpose(0, 1).contiguous()
        h_final = h_final.squeeze(0)  # [B, H]

        # === Output ===
        out = self.to_out(h_seq)

        if return_next_prev_hidden:
            return out, h_final
        return out


class HasteGRU_MultLM(nn.Module):
    """
    Language model using Haste GRU + SiLU Multiplicative Gating.

    EXACT STRUCTURAL COPY of CuDNNGRU_MultLM:
    - Same weight tying (embedding = output projection)
    - Same LayerNorm (not RMSNorm)
    - Same pre-norm residual pattern
    - Same initialization
    - Same loss computation

    Only difference: GRU backend (haste GRU_SiLU vs cuDNN nn.GRU)
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,  # Must be 1.0 for haste
        use_input_gate: bool = True,  # Always True for haste (fused)
        use_hidden_gate: bool = True,  # Always True for haste (fused)
        use_glu_gate: bool = False,  # Not supported
        use_swiglu: bool = False,  # Not supported
        gate_activation: str = 'silu',  # Always silu
        ff_mult: float = 0.0,  # Optional FFN (0 = disabled)
        dropout: float = 0.0,
        tie_weights: bool = True,
        use_checkpointing: bool = False,
        inner_chunk_size: int = 128,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.use_checkpointing = use_checkpointing
        self.inner_chunk_size = inner_chunk_size

        # === Token embedding (IDENTICAL to CuDNNGRU_MultLM) ===
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # === GRU + Multiplicative layers (using Haste backend) ===
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList() if ff_mult > 0 else None
        self.ff_norms = nn.ModuleList() if ff_mult > 0 else None

        for i in range(depth):
            self.layers.append(
                HasteGRU_Mult(
                    dim=dim,
                    expansion_factor=expansion_factor,
                    gate_expansion=gate_expansion,
                    use_input_gate=use_input_gate,
                    use_hidden_gate=use_hidden_gate,
                    use_glu_gate=use_glu_gate,
                    use_swiglu=use_swiglu,
                    gate_activation=gate_activation,
                    layer_idx=i,
                    num_layers=depth,
                )
            )
            # IDENTICAL: LayerNorm (not RMSNorm!)
            self.norms.append(nn.LayerNorm(dim))

            if ff_mult > 0:
                self.ff_layers.append(nn.Sequential(
                    nn.Linear(dim, int(dim * ff_mult), bias=False),
                    nn.GELU(),
                    nn.Linear(int(dim * ff_mult), dim, bias=False),
                ))
                self.ff_norms.append(nn.LayerNorm(dim))

        # === Output (IDENTICAL to CuDNNGRU_MultLM) ===
        self.norm_f = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        # CRITICAL: Weight tying (embedding = output projection)
        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights IDENTICALLY to CuDNNGRU_MultLM."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if self.ff_layers is not None:
            for ff in self.ff_layers:
                nn.init.normal_(ff[0].weight, std=0.02)
                nn.init.normal_(ff[2].weight, std=0.02 / math.sqrt(2 * self.depth))

    def _process_chunk(self, h_chunk, layer_hiddens):
        """Process a single chunk through all layers."""
        new_hiddens = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            prev_h = layer_hiddens[i] if layer_hiddens else None
            layer_out, next_h = layer(norm(h_chunk), prev_hidden=prev_h,
                                       return_next_prev_hidden=True)
            h_chunk = h_chunk + layer_out
            new_hiddens.append(next_h)

            if self.ff_layers is not None:
                h_chunk = h_chunk + self.ff_layers[i](self.ff_norms[i](h_chunk))

        return h_chunk, new_hiddens

    def _checkpointed_chunk_fast(self, h_chunk, packed_hiddens):
        """Fast checkpointing that operates on packed tensors directly."""
        def inner_fn(h_chunk, packed_hiddens):
            new_hiddens = torch.empty_like(packed_hiddens)
            h = h_chunk

            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                prev_h = packed_hiddens[i]
                layer_out, next_h = layer(norm(h), prev_hidden=prev_h,
                                           return_next_prev_hidden=True)
                h = h + layer_out
                new_hiddens[i] = next_h

                if self.ff_layers is not None:
                    h = h + self.ff_layers[i](self.ff_norms[i](h))

            return h, new_hiddens

        return ckpt.checkpoint(inner_fn, h_chunk, packed_hiddens, use_reentrant=False)

    def forward(
        self,
        x,
        prev_hidden=None,
        prev_hiddens=None,
        prev_conv_buffers=None,
        return_next_prev_hidden=False,
        return_prev_hiddens=False,
        return_loss=False,
        doc_boundaries=None,
        actual_length=None,
        **kwargs,
    ):
        """
        Forward pass - EXACT interface match to CuDNNGRU_MultLM.

        Args:
            actual_length: [B] tensor of valid token counts per sequence.
                           If provided, loss is only computed on valid tokens (not padding).

        Returns loss directly (not logits) for training compatibility.
        """
        B, T = x.shape

        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        if self.use_checkpointing and T > self.inner_chunk_size:
            # === Chunked processing with gradient checkpointing ===
            num_chunks = (T + self.inner_chunk_size - 1) // self.inner_chunk_size
            output_buffer = torch.empty(B, T, self.dim, device=h.device, dtype=h.dtype)

            if prev_hiddens:
                packed_hiddens = torch.stack(prev_hiddens, dim=0)
            else:
                packed_hiddens = torch.zeros(
                    self.depth, B, self.layers[0].dim_inner,
                    device=h.device, dtype=h.dtype
                )

            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.inner_chunk_size
                end = min(start + self.inner_chunk_size, T)
                h_chunk = h[:, start:end, :]

                h_out, packed_hiddens = self._checkpointed_chunk_fast(h_chunk, packed_hiddens)
                output_buffer[:, start:end, :] = h_out

            h = output_buffer
            layer_hiddens = [packed_hiddens[i] for i in range(packed_hiddens.size(0))]
        else:
            # === Standard processing (no checkpointing) ===
            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                h = h + layer(norm(h))

                if self.ff_layers is not None:
                    h = h + self.ff_layers[i](self.ff_norms[i](h))

        # Output (IDENTICAL to CuDNNGRU_MultLM)
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token
        # If actual_length provided, only compute loss on valid tokens (not padding)
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()

        if actual_length is not None:
            # Create mask for valid positions (exclude padding)
            # actual_length is the number of valid tokens, so valid positions are [0, actual_length-1]
            # For next-token prediction, we predict positions [1, actual_length-1], so mask [0, actual_length-2]
            # Use ignore_index=-100 for padding positions
            mask = torch.arange(T - 1, device=x.device).unsqueeze(0) < (actual_length.unsqueeze(1) - 1)
            targets_masked = targets.clone()
            targets_masked[~mask] = -100  # Ignore padding in loss
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                targets_masked.view(-1),
                ignore_index=-100,
            )
        else:
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                targets.view(-1),
            )

        if not return_prev_hiddens and not return_next_prev_hidden:
            return loss

        # Return hidden states for TBPTT if requested
        if self.use_checkpointing and T > self.inner_chunk_size:
            return loss, (layer_hiddens, None)
        return loss, (None, None)


def count_parameters(model):
    """Count and breakdown parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if 'token_emb' in n)
    gru = sum(p.numel() for n, p in model.named_parameters() if 'gru' in n)
    gate = sum(p.numel() for n, p in model.named_parameters() if 'gate' in n)
    ff = sum(p.numel() for n, p in model.named_parameters() if 'ff' in n)

    return {
        'total': total,
        'embedding': embedding,
        'gru': gru,
        'gate': gate,
        'ff': ff,
    }


def copy_weights_from_cudnn(haste_model, cudnn_model):
    """
    Copy weights from CuDNNGRU_MultLM to HasteGRU_MultLM with gate permutation.

    The GRU gate order differs between PyTorch cuDNN and haste:
    - PyTorch cuDNN: [reset, update, candidate]
    - Haste: [update, reset, candidate]

    This function handles the permutation automatically.

    Args:
        haste_model: HasteGRU_MultLM instance (destination)
        cudnn_model: CuDNNGRU_MultLM instance (source)
    """
    dim = haste_model.dim
    depth = haste_model.depth
    H = dim  # hidden size = dim when expansion=1.0

    with torch.no_grad():
        # Token embedding (weight-tied to output, so only copy once)
        haste_model.token_emb.weight.copy_(cudnn_model.token_emb.weight)

        # Norms
        for i in range(depth):
            haste_model.norms[i].weight.copy_(cudnn_model.norms[i].weight)
            haste_model.norms[i].bias.copy_(cudnn_model.norms[i].bias)

        # Final norm
        haste_model.norm_f.weight.copy_(cudnn_model.norm_f.weight)
        haste_model.norm_f.bias.copy_(cudnn_model.norm_f.bias)

        # FFN layers (if present)
        if haste_model.ff_layers is not None and cudnn_model.ff_layers is not None:
            for i in range(depth):
                haste_model.ff_layers[i][0].weight.copy_(cudnn_model.ff_layers[i][0].weight)
                haste_model.ff_layers[i][2].weight.copy_(cudnn_model.ff_layers[i][2].weight)
                haste_model.ff_norms[i].weight.copy_(cudnn_model.ff_norms[i].weight)
                haste_model.ff_norms[i].bias.copy_(cudnn_model.ff_norms[i].bias)

        # GRU layers with gate permutation
        for i in range(depth):
            cudnn_layer = cudnn_model.layers[i]
            haste_layer = haste_model.layers[i]

            # GRU input kernel: PyTorch [3H, C] (r,z,n) -> Haste [C, 3H] (z,r,g)
            pt_ih = cudnn_layer.gru.weight_ih_l0
            pt_r, pt_z, pt_n = pt_ih[0:H], pt_ih[H:2*H], pt_ih[2*H:3*H]
            haste_ih = torch.cat([pt_z, pt_r, pt_n], dim=0)
            haste_layer.gru_silu.kernel.copy_(haste_ih.T)

            # GRU recurrent kernel: same permutation
            pt_hh = cudnn_layer.gru.weight_hh_l0
            pt_hr, pt_hz, pt_hn = pt_hh[0:H], pt_hh[H:2*H], pt_hh[2*H:3*H]
            haste_hh = torch.cat([pt_hz, pt_hr, pt_hn], dim=0)
            haste_layer.gru_silu.recurrent_kernel.copy_(haste_hh.T)

            # GRU input bias: same permutation
            pt_bih = cudnn_layer.gru.bias_ih_l0
            haste_layer.gru_silu.bias.copy_(
                torch.cat([pt_bih[H:2*H], pt_bih[0:H], pt_bih[2*H:3*H]])
            )

            # GRU recurrent bias: same permutation
            pt_bhh = cudnn_layer.gru.bias_hh_l0
            haste_layer.gru_silu.recurrent_bias.copy_(
                torch.cat([pt_bhh[H:2*H], pt_bhh[0:H], pt_bhh[2*H:3*H]])
            )

            # Gate weights: transpose (PyTorch nn.Linear [out, in] -> haste [in, out])
            haste_layer.gru_silu.gate_kernel_x.copy_(cudnn_layer.gate_x.weight.T)
            haste_layer.gru_silu.gate_kernel_h.copy_(cudnn_layer.gate_h.weight.T)
            haste_layer.gru_silu.gate_bias.copy_(cudnn_layer.gate_bias)


if __name__ == "__main__":
    print("Testing HasteGRU_Mult...")
    print("=" * 60)

    # Test single layer
    layer = HasteGRU_Mult(
        dim=256,
        expansion_factor=1.0,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Test full LM
    print("\n" + "=" * 60)
    print("Testing HasteGRU_MultLM...")

    model = HasteGRU_MultLM(
        num_tokens=50280,
        dim=2048,
        depth=24,
        expansion_factor=1.0,
        ff_mult=0.0,
    ).cuda().bfloat16()

    counts = count_parameters(model)
    print(f"Model: {counts['total']:,} params ({counts['total']/1e9:.2f}B)")

    x = torch.randint(0, 50280, (2, 256), device='cuda')
    loss = model(x)
    print(f"Forward: loss={loss.item():.4f}")

    loss.backward()
    print("Backward succeeded!")

    # Compare parameter counts with CuDNN version
    print("\n" + "=" * 60)
    print("Comparing with CuDNNGRU_MultLM...")

    from mingru.cudnn_gru_mult import CuDNNGRU_MultLM, count_parameters as count_cudnn

    cudnn_model = CuDNNGRU_MultLM(
        num_tokens=50280,
        dim=2048,
        depth=24,
        expansion_factor=1.0,
        gate_activation='silu',  # Must use silu to match haste
        ff_mult=0.0,
    ).cuda().bfloat16()

    haste_counts = count_parameters(model)
    cudnn_counts = count_cudnn(cudnn_model)

    print(f"HasteGRU_MultLM: {haste_counts['total']:,} params")
    print(f"CuDNNGRU_MultLM: {cudnn_counts['total']:,} params")
    print(f"Difference: {haste_counts['total'] - cudnn_counts['total']:,}")

    print("\nDone!")

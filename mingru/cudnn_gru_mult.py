"""
cuDNN GRU + Multiplicative Gating for nonlinear sequence mixing.

Key insight from nonlinearity research:
- Standard GRUs have element-wise sigmoid/tanh gates
- These gates don't enable XOR/parity reasoning on sequences
- Multiplicative interactions h*f(x) can solve problems linear models cannot

Architecture:
  h = cuDNN_GRU(x)                       # Fast sequence mixing via cuDNN
  gate = σ(W_x @ x + W_h @ h + b)        # Content+state-dependent gate
  h' = h * gate                          # Multiplicative interaction
  out = W_out @ h'                       # Output projection

The multiplicative gate creates second-order terms (h_i * x_j) which
can implement XOR-like operations that pure linear recurrences cannot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import math


class CuDNNGRU_Mult(nn.Module):
    """
    cuDNN GRU with post-hoc multiplicative gating.

    This is the simplest possible test of multiplicative interactions:
    1. Run cuDNN GRU (fast, hardware-accelerated)
    2. Apply multiplicative gate: h' = h * σ(Wx·x + Wh·h)
    3. Project to output

    The multiplicative gate creates interactions between input content
    and hidden state that enable nonlinear temporal reasoning.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,  # How big is the gate hidden dim
        use_input_gate: bool = True,  # Gate depends on input x
        use_hidden_gate: bool = True,  # Gate depends on hidden h
        use_glu_gate: bool = False,  # GLU-style: split h into h1 and gate
        gate_activation: str = 'sigmoid',  # 'sigmoid' or 'silu' (like Mamba2)
        layer_idx: int = None,
        num_layers: int = None,
        **kwargs  # Ignore other params for API compat
    ):
        super().__init__()

        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.gate_dim = int(self.dim_inner * gate_expansion)
        self.use_input_gate = use_input_gate
        self.use_hidden_gate = use_hidden_gate
        self.use_glu_gate = use_glu_gate
        self.gate_activation = gate_activation

        # === cuDNN GRU ===
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        if use_glu_gate:
            # === GLU-style Gate (Option 3) ===
            # Project h to 2x, split, apply GLU: out = h1 * σ(h2)
            # This tests: is x-dependence in gate critical, or is h-dependent enough?
            self.glu_proj = nn.Linear(self.dim_inner, 2 * self.dim_inner, bias=True)
            self.gate_x = None
            self.gate_h = None
            self.gate_bias = None
            self.gate_proj = None
        else:
            # === Multiplicative Gate ===
            # gate = σ(W_x @ x + W_h @ h + b)
            # This creates x-dependent and h-dependent gating
            self.glu_proj = None

            if use_input_gate:
                self.gate_x = nn.Linear(dim, self.gate_dim, bias=False)
            else:
                self.gate_x = None

            if use_hidden_gate:
                self.gate_h = nn.Linear(self.dim_inner, self.gate_dim, bias=False)
            else:
                self.gate_h = None

            self.gate_bias = nn.Parameter(torch.zeros(self.gate_dim))

            # If gate_dim != dim_inner, need projection
            if self.gate_dim != self.dim_inner:
                self.gate_proj = nn.Linear(self.gate_dim, self.dim_inner, bias=False)
            else:
                self.gate_proj = None

        # === Output projection ===
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Initialize gate projections with small values
        # Bias at 0 means gate starts at 0.5 (neutral)
        if self.use_glu_gate:
            nn.init.normal_(self.glu_proj.weight, std=0.02)
            nn.init.zeros_(self.glu_proj.bias)
        else:
            if self.gate_x is not None:
                nn.init.normal_(self.gate_x.weight, std=0.02)
            if self.gate_h is not None:
                nn.init.normal_(self.gate_h.weight, std=0.02)
            if self.gate_proj is not None:
                nn.init.normal_(self.gate_proj.weight, std=0.02)

        # Output projection: small init for residual learning
        if not isinstance(self.to_out, nn.Identity):
            nn.init.normal_(self.to_out.weight, std=0.02 / math.sqrt(2))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        """
        Forward pass.

        Args:
            x: [B, T, D] input
            prev_hidden: [B, dim_inner] previous GRU hidden state
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

        # === cuDNN GRU ===
        # cuDNN GRU needs matching dtypes - use the GRU weight dtype
        gru_dtype = self.gru.weight_ih_l0.dtype
        x_gru = x.to(gru_dtype).contiguous()

        if prev_hidden is not None:
            if prev_hidden.dim() == 2:
                h0 = prev_hidden.to(gru_dtype).unsqueeze(0).contiguous()
            else:
                h0 = prev_hidden.to(gru_dtype).contiguous()
        else:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=gru_dtype)

        # Run GRU
        h_seq, h_final = self.gru(x_gru, h0)
        # h_seq: [B, T, dim_inner]
        # h_final: [1, B, dim_inner]

        h_seq = h_seq.to(dtype)
        h_final = h_final.squeeze(0).to(dtype)

        if self.use_glu_gate:
            # === GLU-style Gate (Option 3) ===
            # Project h to 2x, split, apply GLU: out = h1 * σ(h2)
            proj = self.glu_proj(h_seq)  # [B, T, 2*dim_inner]
            h1, h2 = proj.chunk(2, dim=-1)  # Each: [B, T, dim_inner]
            h_gated = h1 * torch.sigmoid(h2)  # GLU gating
        else:
            # === Multiplicative Gate ===
            # gate = σ(W_x @ x + W_h @ h + b)
            gate_logits = self.gate_bias.view(1, 1, -1)

            if self.gate_x is not None:
                gate_logits = gate_logits + self.gate_x(x)  # [B, T, gate_dim]

            if self.gate_h is not None:
                gate_logits = gate_logits + self.gate_h(h_seq)  # [B, T, gate_dim]

            # Apply gate activation (sigmoid default, silu like Mamba2)
            if self.gate_activation == 'silu':
                gate = F.silu(gate_logits)  # [B, T, gate_dim]
            else:
                gate = torch.sigmoid(gate_logits)  # [B, T, gate_dim]

            # Project gate to match h_seq dim if needed
            if self.gate_proj is not None:
                gate = self.gate_proj(gate)

            # Apply multiplicative gating: h' = h * gate
            # This is the key nonlinear interaction!
            h_gated = h_seq * gate  # [B, T, dim_inner]

        # === Output ===
        out = self.to_out(h_gated)

        if return_next_prev_hidden:
            return out, h_final
        return out


class CuDNNGRU_MultLM(nn.Module):
    """
    Language model using cuDNN GRU + Multiplicative Gating.

    This is a minimal test of whether multiplicative gating after GRU
    improves language modeling compared to standard GRU.

    Gradient Checkpointing:
        When use_checkpointing=True, the sequence is processed in chunks of
        inner_chunk_size tokens. Hidden states are saved at chunk boundaries
        and activations are recomputed during backward pass.

        Memory: O(T/K + K) instead of O(T) where K = inner_chunk_size
        Compute: ~2x forward passes (recompute during backward)
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        expansion_factor: float = 1.0,
        gate_expansion: float = 1.0,
        use_input_gate: bool = True,
        use_hidden_gate: bool = True,
        use_glu_gate: bool = False,  # GLU-style: split h into h1 and gate
        gate_activation: str = 'sigmoid',  # 'sigmoid' or 'silu' (like Mamba2)
        ff_mult: float = 0.0,  # Optional FFN (0 = disabled)
        dropout: float = 0.0,
        tie_weights: bool = True,
        use_checkpointing: bool = False,  # Enable gradient checkpointing
        inner_chunk_size: int = 128,  # Chunk size for checkpointing
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.use_checkpointing = use_checkpointing
        self.inner_chunk_size = inner_chunk_size

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # GRU + Multiplicative layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList() if ff_mult > 0 else None
        self.ff_norms = nn.ModuleList() if ff_mult > 0 else None

        for i in range(depth):
            self.layers.append(
                CuDNNGRU_Mult(
                    dim=dim,
                    expansion_factor=expansion_factor,
                    gate_expansion=gate_expansion,
                    use_input_gate=use_input_gate,
                    use_hidden_gate=use_hidden_gate,
                    use_glu_gate=use_glu_gate,
                    gate_activation=gate_activation,
                    layer_idx=i,
                    num_layers=depth,
                )
            )
            self.norms.append(nn.LayerNorm(dim))

            if ff_mult > 0:
                self.ff_layers.append(nn.Sequential(
                    nn.Linear(dim, int(dim * ff_mult), bias=False),
                    nn.GELU(),
                    nn.Linear(int(dim * ff_mult), dim, bias=False),
                ))
                self.ff_norms.append(nn.LayerNorm(dim))

        # Output
        self.norm_f = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if self.ff_layers is not None:
            for ff in self.ff_layers:
                nn.init.normal_(ff[0].weight, std=0.02)
                nn.init.normal_(ff[2].weight, std=0.02 / math.sqrt(2 * self.depth))

    def _process_chunk(self, h_chunk, layer_hiddens):
        """
        Process a single chunk through all layers.

        Args:
            h_chunk: [B, chunk_len, D] embedded input chunk
            layer_hiddens: list of [B, dim_inner] hidden states per layer

        Returns:
            h_chunk: [B, chunk_len, D] output
            new_hiddens: list of updated hidden states
        """
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

    def _checkpointed_chunk(self, h_chunk, layer_hiddens):
        """Wrapper for checkpointing that handles hidden states."""
        # torch.utils.checkpoint requires all inputs to be tensors
        # Pack hiddens into a single tensor for checkpointing
        if layer_hiddens:
            packed_hiddens = torch.stack(layer_hiddens, dim=0)  # [depth, B, dim_inner]
        else:
            packed_hiddens = None

        def inner_fn(h_chunk, packed_hiddens):
            if packed_hiddens is not None:
                hiddens = [packed_hiddens[i] for i in range(packed_hiddens.size(0))]
            else:
                hiddens = None
            out, new_hiddens = self._process_chunk(h_chunk, hiddens)
            new_packed = torch.stack(new_hiddens, dim=0)
            return out, new_packed

        if packed_hiddens is not None:
            out, new_packed = ckpt.checkpoint(inner_fn, h_chunk, packed_hiddens,
                                               use_reentrant=False)
        else:
            # First chunk - no hidden state yet
            # Create dummy packed_hiddens for checkpoint signature consistency
            B = h_chunk.size(0)
            device = h_chunk.device
            dtype = h_chunk.dtype
            dummy_hiddens = torch.zeros(self.depth, B, self.layers[0].dim_inner,
                                        device=device, dtype=dtype)
            out, new_packed = ckpt.checkpoint(inner_fn, h_chunk, dummy_hiddens,
                                               use_reentrant=False)

        new_hiddens = [new_packed[i] for i in range(new_packed.size(0))]
        return out, new_hiddens

    def _checkpointed_chunk_fast(self, h_chunk, packed_hiddens):
        """
        Fast checkpointing that operates on packed tensors directly.
        Eliminates Python list operations that cause CPU-GPU sync.

        Args:
            h_chunk: [B, chunk_len, D] embedded input chunk
            packed_hiddens: [depth, B, dim_inner] packed hidden states

        Returns:
            h_out: [B, chunk_len, D] output
            new_packed_hiddens: [depth, B, dim_inner] updated hidden states
        """
        def inner_fn(h_chunk, packed_hiddens):
            # Process through all layers using tensor indexing (no Python lists)
            new_hiddens = torch.empty_like(packed_hiddens)
            h = h_chunk

            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                # Get hidden for this layer directly from tensor (no list conversion)
                prev_h = packed_hiddens[i]
                layer_out, next_h = layer(norm(h), prev_hidden=prev_h,
                                           return_next_prev_hidden=True)
                h = h + layer_out
                new_hiddens[i] = next_h

                if self.ff_layers is not None:
                    h = h + self.ff_layers[i](self.ff_norms[i](h))

            return h, new_hiddens

        # Run with checkpointing
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
        **kwargs,
    ):
        """
        Forward pass matching train.py interface.

        When use_checkpointing=True:
            - Processes sequence in chunks of inner_chunk_size
            - Saves hidden states at boundaries, recomputes activations in backward
            - Memory: O(depth * B * D + T/K * B * D) instead of O(T * B * D)
        """
        B, T = x.shape

        # Embed
        h = self.token_emb(x)
        h = self.drop(h)

        if self.use_checkpointing and T > self.inner_chunk_size:
            # === Chunked processing with gradient checkpointing ===
            # Pre-allocate output tensor to avoid Python list append + cat sync
            num_chunks = (T + self.inner_chunk_size - 1) // self.inner_chunk_size
            output_buffer = torch.empty(B, T, self.dim, device=h.device, dtype=h.dtype)

            # Pack layer hiddens as tensor once (not per-chunk)
            # Check for truthy (not None and not empty list)
            if prev_hiddens:
                packed_hiddens = torch.stack(prev_hiddens, dim=0)  # [depth, B, dim_inner]
            else:
                packed_hiddens = torch.zeros(
                    self.depth, B, self.layers[0].dim_inner,
                    device=h.device, dtype=h.dtype
                )

            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.inner_chunk_size
                end = min(start + self.inner_chunk_size, T)
                h_chunk = h[:, start:end, :]

                # Process chunk with checkpointing - pass packed tensor directly
                h_out, packed_hiddens = self._checkpointed_chunk_fast(h_chunk, packed_hiddens)

                # Write directly to pre-allocated buffer (no Python list!)
                output_buffer[:, start:end, :] = h_out

            h = output_buffer
            layer_hiddens = [packed_hiddens[i] for i in range(packed_hiddens.size(0))]
        else:
            # === Standard processing (no checkpointing) ===
            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                h = h + layer(norm(h))

                if self.ff_layers is not None:
                    h = h + self.ff_layers[i](self.ff_norms[i](h))

        # Output
        h = self.norm_f(h)
        logits = self.to_logits(h)

        # Compute loss: predict next token
        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
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


if __name__ == "__main__":
    print("Testing CuDNNGRU_Mult...")
    print("=" * 60)

    # Test single layer
    layer = CuDNNGRU_Mult(
        dim=256,
        expansion_factor=1.0,
        gate_expansion=1.0,
        use_input_gate=True,
        use_hidden_gate=True,
    ).cuda().bfloat16()

    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out, h_final = layer(x, return_next_prev_hidden=True)
    print(f"Layer output shape: {out.shape}")
    print(f"Hidden shape: {h_final.shape}")

    loss = out.sum()
    loss.backward()
    print("Layer backward pass succeeded!")

    # Test full LM - find ~1B config
    print("\n" + "=" * 60)
    print("Finding ~1B params configuration:")

    for depth in [20, 24, 28, 32]:
        model = CuDNNGRU_MultLM(
            num_tokens=50280,
            dim=2048,
            depth=depth,
            expansion_factor=1.0,
            gate_expansion=1.0,
            use_input_gate=True,
            use_hidden_gate=True,
            ff_mult=0.0,  # No FFN for first test
        )

        counts = count_parameters(model)
        print(f"\ndepth={depth}:")
        print(f"  Total: {counts['total']:,} ({counts['total']/1e9:.2f}B)")
        print(f"  GRU: {counts['gru']:,} ({counts['gru']/1e6:.1f}M)")
        print(f"  Gate: {counts['gate']:,} ({counts['gate']/1e6:.1f}M)")

        del model

    # Test forward/backward on small model
    print("\n" + "=" * 60)
    print("Testing forward/backward...")

    model = CuDNNGRU_MultLM(
        num_tokens=50280,
        dim=2048,
        depth=24,  # ~1B params
        expansion_factor=1.0,
        gate_expansion=1.0,
        use_input_gate=True,
        use_hidden_gate=True,
        ff_mult=0.0,
    ).cuda().bfloat16()

    counts = count_parameters(model)
    print(f"Model: {counts['total']:,} params ({counts['total']/1e9:.2f}B)")

    x = torch.randint(0, 50280, (2, 256), device='cuda')
    loss = model(x)
    print(f"Forward: loss={loss.item():.4f}")

    loss.backward()
    print("Backward succeeded!")

    # Test gradient checkpointing
    print("\n" + "=" * 60)
    print("Testing gradient checkpointing...")

    # Small model for checkpointing test
    model_ckpt = CuDNNGRU_MultLM(
        num_tokens=50280,
        dim=512,
        depth=8,
        expansion_factor=1.0,
        gate_expansion=1.0,
        use_input_gate=True,
        use_hidden_gate=True,
        ff_mult=0.0,
        use_checkpointing=True,
        inner_chunk_size=64,  # Small chunks for testing
    ).cuda().bfloat16()

    counts = count_parameters(model_ckpt)
    print(f"Checkpointed model: {counts['total']:,} params")

    # Test with sequence > inner_chunk_size to trigger checkpointing
    x = torch.randint(0, 50280, (2, 256), device='cuda')
    loss = model_ckpt(x)
    print(f"Forward (checkpointed): loss={loss.item():.4f}")

    loss.backward()
    print("Backward (checkpointed) succeeded!")

    # Verify outputs match between checkpointed and non-checkpointed
    print("\n" + "=" * 60)
    print("Verifying checkpointed vs non-checkpointed outputs match...")

    model_no_ckpt = CuDNNGRU_MultLM(
        num_tokens=50280,
        dim=512,
        depth=8,
        expansion_factor=1.0,
        gate_expansion=1.0,
        use_input_gate=True,
        use_hidden_gate=True,
        ff_mult=0.0,
        use_checkpointing=False,  # No checkpointing
    ).cuda().bfloat16()

    # Copy weights
    model_no_ckpt.load_state_dict(model_ckpt.state_dict())

    # Same input
    torch.manual_seed(42)
    x = torch.randint(0, 50280, (2, 256), device='cuda')

    loss_ckpt = model_ckpt(x)
    loss_no_ckpt = model_no_ckpt(x)

    diff = abs(loss_ckpt.item() - loss_no_ckpt.item())
    print(f"Loss (checkpointed): {loss_ckpt.item():.6f}")
    print(f"Loss (no checkpoint): {loss_no_ckpt.item():.6f}")
    print(f"Difference: {diff:.6f}")
    assert diff < 1e-3, f"Loss mismatch! diff={diff}"
    print("Verification passed!")

    print("\nDone!")

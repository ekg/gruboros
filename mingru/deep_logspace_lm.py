"""
Deep Log-Space GRU Language Model with Mamba-style architecture.

Key features:
- Log-space hidden states for numerical stability
- Residual connections between layers (like Mamba)
- Layer normalization before each GRU layer (pre-norm)
- Designed for deep networks (20-32 layers)
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from mingru.logspace_hybrid_gru import LogSpaceHybridGRU


class DeepLogSpaceGRULM(Module):
    """
    Deep GRU Language Model with log-space numerics and Mamba-style architecture.

    Architecture per layer:
    ```
    x → LayerNorm → LogSpaceGRU → (+) → next layer
        ↓_________________________↑
              (residual)
    ```

    This is based on Mamba's pre-normalization + residual design,
    adapted for recurrent GRUs with log-space hidden states.
    """

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        expansion = 1.5,
        dropout = 0.0,
        z_bias_input = -2.0,
        z_bias_hidden = -2.0,
        recurrence_chunk_size = 64
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.expansion = expansion

        # Token embedding
        self.token_emb = nn.Embedding(num_tokens, dim)

        # GRU layers with pre-normalization and residuals
        self.gru_layers = ModuleList([])
        self.layer_norms = ModuleList([])
        self.dropouts = ModuleList([])

        for i in range(depth):
            # Pre-normalization (before GRU)
            self.layer_norms.append(nn.LayerNorm(dim))

            # Log-space GRU
            self.gru_layers.append(
                LogSpaceHybridGRU(
                    dim=dim,
                    expansion_factor=expansion,
                    z_bias_input=z_bias_input,
                    z_bias_hidden=z_bias_hidden,
                    recurrence_chunk_size=recurrence_chunk_size
                )
            )

            # Optional dropout after residual
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(None)

        # Final normalization before output
        self.final_norm = nn.LayerNorm(dim)

        # Output projection
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        # Tie embedding and output weights (standard LM practice)
        self.to_logits.weight = self.token_emb.weight

        print(f"\n=== DeepLogSpaceGRULM ===")
        print(f"Depth: {depth} layers")
        print(f"Dim: {dim}")
        print(f"Expansion: {expansion}")
        print(f"Inner dim: {int(dim * expansion)}")
        print(f"Dropout: {dropout}")
        print(f"Architecture: LayerNorm → LogSpaceGRU → Residual × {depth}")
        print(f"=========================\n")

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_hiddens = False,
        prev_hiddens = None,
        prev_conv_buffers = None,  # Not used, for API compatibility
        actual_length = None,
        doc_boundaries = None
    ):
        """
        Forward pass through deep log-space GRU layers.

        Args:
            x: [B, T] token indices
            return_loss: bool - compute cross-entropy loss
            return_prev_hiddens: bool - return hidden states
            prev_hiddens: List[Tensor] - previous log-space hidden states (or None)
            doc_boundaries: [B, T] bool - document boundary resets
            actual_length: [B] - actual sequence lengths (for masking)

        Returns:
            If return_loss:
                loss: scalar
                (next_hiddens, None): tuple for API compatibility
            Else:
                logits: [B, T, V]
                (next_hiddens, None): tuple (if return_prev_hiddens)
        """
        B, T = x.shape
        device = x.device

        # Save input tokens for loss computation
        input_tokens = x

        # Embed tokens
        x = self.token_emb(x)  # [B, T, D]

        # Initialize hidden states if needed
        if prev_hiddens is None:
            prev_hiddens = [None] * self.depth

        next_hiddens = []

        # Pass through GRU layers with residual connections
        for i, (ln, gru, dropout) in enumerate(zip(
            self.layer_norms, self.gru_layers, self.dropouts
        )):
            # Save input for residual
            residual = x

            # Pre-normalization
            x_norm = ln(x)

            # GRU forward
            x_out, log_h = gru(
                x_norm,
                prev_hidden=prev_hiddens[i],
                return_next_prev_hidden=True,
                doc_boundaries=doc_boundaries
            )

            # Residual connection (KEY for deep networks!)
            x = residual + x_out

            # Optional dropout
            if dropout is not None:
                x = dropout(x)

            # Save hidden state for next iteration
            next_hiddens.append(log_h)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.to_logits(x)  # [B, T, V]

        if return_loss:
            # Compute cross-entropy loss
            # Shift logits and targets for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_tokens[:, 1:].contiguous()

            # Flatten for loss computation
            logits_flat = shift_logits.reshape(-1, self.num_tokens)
            targets_flat = shift_targets.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction='mean'
            )

            if return_prev_hiddens:
                return loss, (next_hiddens, None)  # None for conv_buffers compatibility
            return loss

        # Return logits (inference mode)
        if return_prev_hiddens:
            return logits, (next_hiddens, None)
        return logits

    def count_parameters(self):
        """Count total parameters in model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n=== Parameter Count ===")
        print(f"Total: {total:,} ({total/1e6:.1f}M)")
        print(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M)")

        # Break down by component
        emb_params = self.token_emb.weight.numel()
        gru_params = sum(p.numel() for layer in self.gru_layers for p in layer.parameters())
        ln_params = sum(p.numel() for ln in self.layer_norms for p in ln.parameters())
        ln_params += self.final_norm.weight.numel() + self.final_norm.bias.numel()

        print(f"  Embedding: {emb_params:,} ({emb_params/1e6:.1f}M)")
        print(f"  GRU layers: {gru_params:,} ({gru_params/1e6:.1f}M)")
        print(f"  LayerNorms: {ln_params:,} ({ln_params/1e6:.1f}M)")
        print(f"  Output: {self.to_logits.weight.numel():,} ({self.to_logits.weight.numel()/1e6:.1f}M) [tied]")
        print(f"=======================\n")

        return total


if __name__ == "__main__":
    print("Testing DeepLogSpaceGRULM...")

    # Test with small model first
    model = DeepLogSpaceGRULM(
        num_tokens=50281,  # TikToken vocab size
        dim=512,
        depth=3,
        expansion=1.5,
        dropout=0.1
    ).cuda()

    model.count_parameters()

    # Test forward pass
    B, T = 4, 64
    x = torch.randint(0, 50281, (B, T), device='cuda')

    # Test training mode (with loss)
    loss, (hiddens, _) = model(x, return_loss=True, return_prev_hiddens=True)
    print(f"✓ Training forward pass successful!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Hidden states: {len(hiddens)} layers")
    print(f"Hidden shape: {hiddens[0].shape}")
    print(f"Hidden range: [{hiddens[0].min():.2f}, {hiddens[0].max():.2f}]")

    # Test backward pass
    loss.backward()
    print(f"✓ Backward pass successful!")

    # Check gradients at different layers
    for i, layer in enumerate(model.gru_layers):
        if layer.input_projection.weight.grad is not None:
            grad_norm = layer.input_projection.weight.grad.norm()
            print(f"  Layer {i}: grad_norm = {grad_norm:.4f}")

    # Test inference mode
    with torch.no_grad():
        logits = model(x, return_loss=False)
        print(f"\n✓ Inference forward pass successful!")
        print(f"Logits shape: {logits.shape}")

    print("\n✅ All tests passed!")

    # Test with target 1B configuration
    print("\n" + "="*60)
    print("Testing 1B parameter configuration...")
    print("="*60)

    model_1b = DeepLogSpaceGRULM(
        num_tokens=50281,
        dim=1920,
        depth=24,
        expansion=1.5,
        dropout=0.2
    ).cuda()

    model_1b.count_parameters()

    # Quick forward test
    x_test = torch.randint(0, 50281, (2, 32), device='cuda')
    loss_test = model_1b(x_test, return_loss=True)
    print(f"✓ 1B model forward pass: loss = {loss_test.item():.4f}")

    print("\n✅ 1B model ready for training!")

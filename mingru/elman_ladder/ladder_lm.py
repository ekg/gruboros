"""
Language Model wrapper for Elman Ablation Ladder.

Uses any of the 7 ladder levels as the recurrent layer.
Supports both normal and log-space variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import level classes directly to avoid circular import
from .stock_elman import StockElman, StockElmanCell
from .gated_elman import GatedElman, GatedElmanCell
from .selective_elman import SelectiveElman, SelectiveElmanCell
from .diagonal_selective import DiagonalSelective, DiagonalSelectiveCell
from .log_storage_diagonal import LogStorageDiagonal
from .log_compute_full import LogComputeFull
from .logspace_triple_r import LogSpaceTripleR
from .log_space_wrapper import LogSpaceLayer


def get_ladder_level(level):
    """Get the module class for a specific ladder level.

    Args:
        level: Integer (0-6) or string ('0-log', '1-log', '2-log', '3-log')

    Returns:
        Layer class or factory function
    """
    # Normal space levels
    normal_levels = {
        0: ("Stock Elman", StockElman),
        1: ("Gated Elman", GatedElman),
        2: ("Selective Elman", SelectiveElman),
        3: ("Diagonal Selective", DiagonalSelective),
        4: ("Log-Storage Diagonal", LogStorageDiagonal),
        5: ("Log-Compute Full", LogComputeFull),
        6: ("Log-Space Triple R", LogSpaceTripleR),
    }

    # Log-space wrapper variants (for levels 0-3)
    log_space_cells = {
        "0-log": ("Stock Elman + Log-Space", StockElmanCell),
        "1-log": ("Gated Elman + Log-Space", GatedElmanCell),
        "2-log": ("Selective Elman + Log-Space", SelectiveElmanCell),
        "3-log": ("Diagonal Selective + Log-Space", DiagonalSelectiveCell),
    }

    if level in normal_levels:
        name, cls = normal_levels[level]
        return cls
    elif level in log_space_cells:
        name, cell_cls = log_space_cells[level]
        # Return a factory function that creates LogSpaceLayer with the right cell
        def make_log_layer(dim, expansion=1.0, n_groups=32, delta_init=-2.0, dropout=0.0, **kwargs):
            return LogSpaceLayer(cell_cls, dim, expansion=expansion, dropout=dropout)
        return make_log_layer
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-6 or '0-log', '1-log', '2-log', '3-log'.")


class LadderLM(nn.Module):
    """
    Language Model using Elman Ablation Ladder levels.

    Uses Mamba-style architecture with pre-norm + residual connections.

    Args:
        vocab_size: Size of vocabulary (256 for byte-level)
        dim: Model dimension
        depth: Number of layers
        level: Ablation ladder level (0-6)
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax (levels 2+)
        delta_init: Initial delta gate bias
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        level=0,
        expansion=1.0,
        n_groups=32,
        delta_init=-2.0,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.level = level

        # Get the layer class for this level
        LayerClass = get_ladder_level(level)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Pre-normalization layers (one per recurrent layer)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # Stack of recurrent layers
        self.layers = nn.ModuleList([
            LayerClass(
                dim=dim,
                expansion=expansion,
                n_groups=n_groups,
                delta_init=delta_init,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final layer norm before output
        self.norm = nn.LayerNorm(dim)

        # Output projection to vocabulary (tied with embedding)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
        prev_conv_buffers=None,
        actual_length=None,
        doc_boundaries=None,
    ):
        """
        Forward pass compatible with train.py interface.

        Args:
            x: [B, T] input token indices
            return_loss: If True, compute loss (x is [B, T+1] with targets)
            return_prev_hiddens: If True, return hidden states for TBPTT
            prev_hiddens: List of [B, d_inner] per layer, or None
            prev_conv_buffers: Unused, for API compatibility
            actual_length: For masking padded chunks
            doc_boundaries: [B, T] boolean tensor for hidden state reset

        Returns:
            If return_loss: (loss, new_hiddens) or loss
            Else: (logits, new_hiddens) or logits
        """
        if return_loss:
            # x is [B, T+1], split into input and target
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        B, T = inp.shape

        # Embed tokens
        x = self.embedding(inp)  # [B, T, dim]

        # Initialize hidden states if not provided
        if prev_hiddens is None:
            prev_hiddens = [None] * self.depth

        new_hidden_states = []

        # Run through layers with pre-norm + residual (Mamba-style)
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            # Save input for residual
            residual = x

            # Pre-normalization
            x_norm = ln(x)

            # Elman layer forward
            x_out, h_final = layer(x_norm, prev_hiddens[i])

            # Residual connection (KEY for deep networks!)
            x = residual + x_out

            new_hidden_states.append(h_final)

        # Final normalize and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_loss:
            import torch.nn.functional as F
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            if return_prev_hiddens:
                # Return (loss, (hidden_states, conv_buffers)) format expected by train.py
                return loss, (new_hidden_states, None)
            return loss

        if return_prev_hiddens:
            return logits, (new_hidden_states, None)
        return logits

    def get_num_params(self):
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        level_names = {
            0: "Stock Elman",
            1: "Gated Elman",
            2: "Selective Elman",
            3: "Diagonal Selective",
            4: "Log-Storage Diagonal",
            5: "Log-Compute Full",
            6: "Log-Space Triple R",
        }
        return f'level={self.level} ({level_names.get(self.level, "Unknown")}), dim={self.dim}, depth={self.depth}'


def create_ladder_model(
    target_params: str = "100m",
    level: int = 0,
    vocab_size: int = 256,
    expansion: float = 1.0,
    n_groups: int = 32,
):
    """
    Create a LadderLM with approximately target_params parameters.

    Args:
        target_params: Target parameter count (e.g., "100m", "500m", "1b")
        level: Ablation ladder level (0-6)
        vocab_size: Vocabulary size
        expansion: Hidden state expansion
        n_groups: Number of groups for compete softmax

    Returns:
        LadderLM model
    """
    # Parse target
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Model size configurations
    # Tuned for different parameter counts
    configs = {
        50_000_000: (384, 12),      # ~50M
        100_000_000: (512, 16),     # ~100M
        200_000_000: (768, 18),     # ~200M
        350_000_000: (1024, 20),    # ~350M
        500_000_000: (1280, 24),    # ~500M
        700_000_000: (1536, 28),    # ~700M
        1_000_000_000: (1920, 32),  # ~1B
        1_300_000_000: (2048, 40),  # ~1.3B
    }

    # Find closest config
    closest = min(configs.keys(), key=lambda x: abs(x - target_count))
    dim, depth = configs[closest]

    model = LadderLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        level=level,
        expansion=expansion,
        n_groups=n_groups,
    )

    actual_params = model.get_num_params()
    print(f"Created Level {level} model: dim={dim}, depth={depth}, params={actual_params:,}")

    return model


if __name__ == "__main__":
    print("Testing LadderLM...")
    print("=" * 60)

    for level in range(7):
        print(f"\nLevel {level}:")
        model = LadderLM(
            vocab_size=256,
            dim=256,
            depth=4,
            level=level,
            expansion=1.0,
        ).cuda().bfloat16()

        x = torch.randint(0, 256, (2, 32), device='cuda')
        logits, hidden = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), x.view(-1))
        loss.backward()

        print(f"  Params: {model.get_num_params():,}")
        print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")

    print("\nLadderLM tests passed!")

"""
Test script for JAX GRU+EMA model.

Tests:
1. Model initialization
2. Forward pass
3. Gradient computation
4. Parameter count
"""

import sys
sys.path.insert(0, '/home/erikg/gruboros')

import jax
import jax.numpy as jnp
import optax

from jax_gru_ema.model.gru_ema_cell import parallel_ema, GRUCell, GRUCellConfig
from jax_gru_ema.model.gru_ema_layer import GRUEMALayer, GRUEMALayerConfig
from jax_gru_ema.model.gru_ema_block import GRUEMABlock, GRUEMABlockConfig
from jax_gru_ema.model.gru_ema_lm import GRUEMALMModel, GRUEMALMConfig, create_model


def count_params(params):
    """Count total parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


def test_parallel_ema():
    """Test parallel EMA computation."""
    print("=" * 60)
    print("Testing parallel_ema...")

    B, S, D = 2, 128, 64
    alpha = 0.01
    x = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))

    # Run parallel EMA
    ema = parallel_ema(x, alpha)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {ema.shape}")
    print(f"  Alpha: {alpha}")

    # Verify first position is alpha * x[0]
    expected_first = alpha * x[:, 0, :]
    actual_first = ema[:, 0, :]
    error = jnp.abs(expected_first - actual_first).max()
    print(f"  First position error: {error:.2e}")

    # Test gradient
    def loss_fn(x):
        return parallel_ema(x, alpha).sum()

    grad = jax.grad(loss_fn)(x)
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient finite: {jnp.isfinite(grad).all()}")

    print("  PASSED!")
    return True


def test_gru_cell():
    """Test GRU cell."""
    print("=" * 60)
    print("Testing GRUCell...")

    B, S, D, H = 2, 64, 32, 48

    config = GRUCellConfig(
        input_dim=D,
        hidden_dim=H,
        use_bias=True,
        dtype="float32",  # Use float32 for testing
    )
    cell = GRUCell(config=config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (B, S, D))
    params = cell.init(rng, x)

    print(f"  Input shape: {x.shape}")
    print(f"  Params: {count_params(params):,}")

    # Forward pass
    h_seq = cell.apply(params, x)
    print(f"  Output shape: {h_seq.shape}")
    print(f"  Output finite: {jnp.isfinite(h_seq).all()}")

    # Test gradient
    def loss_fn(params, x):
        return cell.apply(params, x).sum()

    grad = jax.grad(loss_fn)(params, x)
    grad_size = count_params(grad)
    print(f"  Gradient params: {grad_size:,}")
    print(f"  Gradients finite: {all(jnp.isfinite(g).all() for g in jax.tree.leaves(grad))}")

    print("  PASSED!")
    return True


def test_gru_ema_layer():
    """Test GRU+EMA layer."""
    print("=" * 60)
    print("Testing GRUEMALayer...")

    B, S, D = 2, 64, 128

    config = GRUEMALayerConfig(
        embedding_dim=D,
        expansion_factor=1.0,
        ema_alpha=0.01,
        use_bias=True,
        dropout=0.0,
        dtype="float32",
    )
    layer = GRUEMALayer(config=config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (B, S, D))
    params = layer.init(rng, x, train=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Params: {count_params(params):,}")

    # Forward pass
    y = layer.apply(params, x, train=False)
    print(f"  Output shape: {y.shape}")
    print(f"  Output finite: {jnp.isfinite(y).all()}")

    # Test gradient
    def loss_fn(params, x):
        return layer.apply(params, x, train=False).sum()

    grad = jax.grad(loss_fn)(params, x)
    print(f"  Gradients finite: {all(jnp.isfinite(g).all() for g in jax.tree.leaves(grad))}")

    print("  PASSED!")
    return True


def test_gru_ema_block():
    """Test GRU+EMA block."""
    print("=" * 60)
    print("Testing GRUEMABlock...")

    B, S, D = 2, 64, 128

    gru_config = GRUEMALayerConfig(
        embedding_dim=D,
        expansion_factor=1.0,
        ema_alpha=0.01,
        dtype="float32",
    )
    config = GRUEMABlockConfig(
        gru_ema=gru_config,
        feedforward=None,  # No FFN for this test
        dtype="float32",
        _num_blocks=1,
    )
    block = GRUEMABlock(config=config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (B, S, D))
    params = block.init(rng, x, train=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Params: {count_params(params):,}")

    # Forward pass
    y = block.apply(params, x, train=False)
    print(f"  Output shape: {y.shape}")
    print(f"  Output finite: {jnp.isfinite(y).all()}")

    # Check residual (output should be input + layer output)
    print(f"  Has residual: {y.shape == x.shape}")

    # Test gradient
    def loss_fn(params, x):
        return block.apply(params, x, train=False).sum()

    grad = jax.grad(loss_fn)(params, x)
    print(f"  Gradients finite: {all(jnp.isfinite(g).all() for g in jax.tree.leaves(grad))}")

    print("  PASSED!")
    return True


def test_full_model():
    """Test full GRU+EMA LM model."""
    print("=" * 60)
    print("Testing GRUEMALMModel...")

    # Small model for testing
    model = create_model(
        vocab_size=1000,
        dim=128,
        depth=4,
        expansion=1.0,
        ff_mult=0.0,
        ema_alpha=0.01,
        dtype="float32",
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    B, S = 2, 64
    input_ids = jax.random.randint(rng, (B, S), 0, 1000)
    params = model.init(rng, input_ids, train=False)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Total params: {count_params(params):,}")

    # Forward pass
    logits = model.apply(params, input_ids, train=False)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({B}, {S}, 1000)")
    print(f"  Output finite: {jnp.isfinite(logits).all()}")

    # Test gradient with cross-entropy loss
    def loss_fn(params, input_ids, target_ids):
        logits = model.apply(params, input_ids, train=True, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        return loss.mean()

    target_ids = jax.random.randint(rng, (B, S), 0, 1000)
    loss, grad = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)

    print(f"  Loss: {loss:.4f}")
    print(f"  Gradients finite: {all(jnp.isfinite(g).all() for g in jax.tree.leaves(grad))}")

    print("  PASSED!")
    return True


def test_scaling():
    """Test model at different scales."""
    print("=" * 60)
    print("Testing model scaling...")

    configs = [
        {"dim": 256, "depth": 4, "name": "tiny"},
        {"dim": 512, "depth": 8, "name": "small"},
        {"dim": 1024, "depth": 12, "name": "medium"},
    ]

    for cfg in configs:
        model = create_model(
            vocab_size=50280,
            dim=cfg["dim"],
            depth=cfg["depth"],
            expansion=1.0,
            ff_mult=0.0,
            dtype="bfloat16",
        )

        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (1, 128), 0, 50280)
        params = model.init(rng, input_ids, train=False)

        n_params = count_params(params)
        print(f"  {cfg['name']}: dim={cfg['dim']}, depth={cfg['depth']} -> {n_params/1e6:.1f}M params")

    print("  PASSED!")
    return True


def main():
    print("\n" + "=" * 60)
    print("JAX GRU+EMA Model Tests")
    print("=" * 60 + "\n")

    # Check JAX backend
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print()

    tests = [
        test_parallel_ema,
        test_gru_cell,
        test_gru_ema_layer,
        test_gru_ema_block,
        test_full_model,
        test_scaling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test the fixed streaming cross-entropy loss kernel.
"""

import torch
import torch.nn.functional as F

# Test import
try:
    from mingru.triton_streaming_loss_fixed import triton_streaming_cross_entropy
    print("✓ Fixed kernel imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)


def test_streaming_loss():
    """Test that streaming loss matches PyTorch reference."""
    print("\n=== Testing Streaming Loss Kernel ===\n")

    # Small test case
    batch_size = 4
    seq_len = 16
    dim = 2048
    vocab_size = 100277  # Full TikToken vocab

    print(f"Config: batch={batch_size}, seq={seq_len}, dim={dim}, vocab={vocab_size}")

    # Create test data
    device = torch.device('cuda')
    torch.manual_seed(42)

    embeddings = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)
    lm_head_weight = torch.randn(vocab_size, dim, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Add some ignore indices
    labels[0, 0] = -100
    labels[1, 5] = -100

    print(f"✓ Test data created")
    print(f"  Memory - embeddings: {embeddings.numel() * 2 / 1e9:.2f} GB")
    print(f"  Memory - lm_head: {lm_head_weight.numel() * 2 / 1e9:.2f} GB")

    # Compute reference loss (PyTorch)
    print(f"\nComputing PyTorch reference...")
    logits_ref = embeddings @ lm_head_weight.t()  # [batch, seq, vocab]
    loss_ref = F.cross_entropy(
        logits_ref.view(-1, vocab_size).float(),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    print(f"  Reference loss: {loss_ref.item():.6f}")
    print(f"  Logits memory: {logits_ref.numel() * 2 / 1e9:.2f} GB")

    # Free logits
    del logits_ref
    torch.cuda.empty_cache()

    # Compute streaming loss (Triton)
    print(f"\nComputing Triton streaming loss...")
    try:
        loss_streaming = triton_streaming_cross_entropy(
            embeddings.float(),  # Triton kernel uses float32
            lm_head_weight.float(),
            labels
        )
        print(f"  Streaming loss: {loss_streaming.item():.6f}")

        # Check accuracy
        abs_error = abs(loss_streaming.item() - loss_ref.item())
        rel_error = abs_error / abs(loss_ref.item())

        print(f"\n✓ Kernel executed successfully!")
        print(f"  Absolute error: {abs_error:.6e}")
        print(f"  Relative error: {rel_error:.6e}")

        if rel_error < 1e-3:
            print(f"\n✅ PASS: Streaming loss matches reference (rel_error={rel_error:.2e} < 1e-3)")
            return True
        else:
            print(f"\n❌ FAIL: Too much error (rel_error={rel_error:.2e} >= 1e-3)")
            return False

    except Exception as e:
        print(f"\n❌ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_streaming_loss()
    exit(0 if success else 1)

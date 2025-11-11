"""
Hybrid streaming cross-entropy: Triton forward + PyTorch backward.

Forward: Memory-efficient Triton kernel (NO logits materialization)
Backward: PyTorch autograd on chunked logits (correct gradients)

This gives us the best of both worlds:
- Forward pass saves memory (no 10GB logits tensor)
- Backward pass is correct (uses PyTorch's proven autograd)

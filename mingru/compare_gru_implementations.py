"""
Compare the original NAU_GRU with the FixedGRU to demonstrate the issues.
"""
import torch
import torch.nn as nn
import time
import sys
import os

# Add mingru to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from nau_gru_cell import NAU_GRU
    has_nau_gru = True
except ImportError:
    print("NAU_GRU not available")
    has_nau_gru = False

from fixed_gru import FixedGRU


def analyze_gradient_flow(model, name, x, steps=5):
    """Analyze gradient flow through multiple steps"""
    print(f"\n=== {name} Gradient Analysis ===")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    grad_norms = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Simple language modeling loss
        output = model(x)
        # Shift for next-token prediction
        loss = F.mse_loss(output[:, :-1], x[:, 1:])
        loss.backward()
        
        # Compute gradient norms
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        losses.append(loss.item())
        grad_norms.append(total_norm)
        optimizer.step()
        
        print(f"Step {step+1}: loss={loss.item():.6f}, grad_norm={total_norm:.6f}")
    
    return losses, grad_norms


def compare_implementations():
    """Compare original NAU_GRU vs FixedGRU"""
    print("Comparing GRU Implementations")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    seq_len = 32
    dim = 64
    
    # Create test data - realistic language modeling scenario
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, dim) * 0.1  # Small inputs like embeddings
    
    print(f"Test setup: batch={batch_size}, seq_len={seq_len}, dim={dim}")
    
    # Test FixedGRU
    print("\n" + "=" * 20 + " FixedGRU " + "=" * 20)
    fixed_gru = FixedGRU(dim=dim, expansion_factor=1.5)
    
    # Forward pass timing
    start = time.time()
    with torch.no_grad():
        fixed_output = fixed_gru(x)
    fixed_time = time.time() - start
    
    print(f"Forward pass time: {fixed_time:.4f}s")
    print(f"Output stats: mean={fixed_output.mean():.4f}, std={fixed_output.std():.4f}")
    print(f"Output range: [{fixed_output.min():.4f}, {fixed_output.max():.4f}]")
    
    # Gradient analysis
    fixed_losses, fixed_grad_norms = analyze_gradient_flow(fixed_gru, "FixedGRU", x.clone())
    
    # Compare with NAU_GRU if available
    if has_nau_gru:
        print("\n" + "=" * 20 + " NAU_GRU " + "=" * 20)
        try:
            nau_gru = NAU_GRU(dim=dim, expansion_factor=1.5)
            
            # Forward pass timing
            start = time.time()
            with torch.no_grad():
                nau_output = nau_gru(x)
            nau_time = time.time() - start
            
            print(f"Forward pass time: {nau_time:.4f}s")
            print(f"Output stats: mean={nau_output.mean():.4f}, std={nau_output.std():.4f}")
            print(f"Output range: [{nau_output.min():.4f}, {nau_output.max():.4f}]")
            
            # Gradient analysis
            nau_losses, nau_grad_norms = analyze_gradient_flow(nau_gru, "NAU_GRU", x.clone())
            
            # Compare performance
            print(f"\n" + "=" * 20 + " Comparison " + "=" * 20)
            print(f"Speedup: {nau_time/fixed_time:.2f}x (NAU_GRU vs FixedGRU)")
            
            # Compare gradient stability
            avg_fixed_grad = sum(fixed_grad_norms) / len(fixed_grad_norms)
            avg_nau_grad = sum(nau_grad_norms) / len(nau_grad_norms)
            print(f"Avg gradient norm - FixedGRU: {avg_fixed_grad:.4f}, NAU_GRU: {avg_nau_grad:.4f}")
            
        except Exception as e:
            print(f"Error testing NAU_GRU: {e}")
    
    # Test numerical stability with extreme inputs
    print(f"\n" + "=" * 20 + " Stability Test " + "=" * 20)
    x_extreme = torch.randn(2, 16, dim) * 10.0  # Large inputs
    
    print("Testing with large inputs (10x normal)...")
    
    try:
        fixed_extreme = fixed_gru(x_extreme)
        if torch.isfinite(fixed_extreme).all():
            print("✓ FixedGRU: stable with large inputs")
        else:
            print("✗ FixedGRU: produces NaN/Inf")
    except Exception as e:
        print(f"✗ FixedGRU: fails with large inputs: {e}")
    
    if has_nau_gru:
        try:
            nau_extreme = nau_gru(x_extreme)
            if torch.isfinite(nau_extreme).all():
                print("✓ NAU_GRU: stable with large inputs")
            else:
                print("✗ NAU_GRU: produces NaN/Inf")
        except Exception as e:
            print(f"✗ NAU_GRU: fails with large inputs: {e}")


def explain_the_problems():
    """Explain the specific issues with NAU_GRU"""
    print("\nNAU_GRU Issues Identified:")
    print("=" * 40)
    
    print("1. ❌ WRONG MATHEMATICS")
    print("   - Uses minGRU's log_g activation instead of proper GRU gates")
    print("   - Missing reset gate, update gate, and tanh candidate")
    print("   - Result: Not actually a GRU, just a weird RNN variant")
    
    print("\n2. ❌ PRECISION PROBLEMS")
    print("   - Converts to float32 in Triton, back to original dtype")
    print("   - Accumulates errors over long sequences (128+ timesteps)")
    print("   - Especially bad with bfloat16")
    
    print("\n3. ❌ GRADIENT FLOW ISSUES")
    print("   - No residual connection (fixed_output + x)")
    print("   - exp() operations with bad gradients")
    print("   - Extreme initialization (-20.0) causes vanishing gradients")
    
    print("\n4. ❌ BATCH SIZE HANDLING")  
    print("   - Resets to -20.0 when batch size changes")
    print("   - Causes training instability during validation")
    
    print("\n5. ❌ TRITON KERNEL BUGS")
    print("   - BLOCK_SIZE chunking issues when dim_inner not divisible")
    print("   - Mask handling problems")
    
    print("\nFixedGRU Solutions:")
    print("=" * 20)
    print("✅ Proper GRU mathematics with 3 gates")
    print("✅ No dtype conversions - stays in original precision")
    print("✅ Residual connections for gradient flow") 
    print("✅ Reasonable initialization (zeros, not -20.0)")
    print("✅ Graceful batch size handling")
    print("✅ Numerically stable operations")


if __name__ == "__main__":
    import torch.nn.functional as F
    explain_the_problems()
    compare_implementations()
#!/usr/bin/env python3
"""
Quick test script to compare FixedGRU vs original NAU_GRU performance.
Run this before doing full training to verify the fix works.
"""
import torch
import torch.nn.functional as F
import time
import sys
import os

# Set up environment
sys.path.insert(0, 'mingru')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use single GPU for test

def create_synthetic_data(batch_size=8, seq_len=128, vocab_size=256, num_batches=100):
    """Create synthetic byte-level language modeling data"""
    print(f"Creating synthetic data: {num_batches} batches of {batch_size}x{seq_len}")
    
    # Create realistic byte sequences (not random)
    data = []
    for _ in range(num_batches):
        # Generate somewhat structured byte sequences
        batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        
        # Add some structure (repeated patterns, etc.)
        for b in range(batch_size):
            # Add some repeated tokens to make it more language-like
            pattern = torch.randint(65, 90, (4,))  # A-Z range
            for i in range(0, seq_len - 4, 8):
                if i + 4 < seq_len:
                    batch[b, i:i+4] = pattern
        
        data.append(batch)
    
    return data

def test_model_on_synthetic_data(model_name, create_model_fn, data, device='cuda'):
    """Test a model on synthetic language modeling data"""
    print(f"\n{'='*20} Testing {model_name} {'='*20}")
    
    model = create_model_fn().to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Track metrics
    losses = []
    times = []
    
    print("Training for 10 steps...")
    for step in range(min(10, len(data))):
        batch = data[step].to(device)
        
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Language modeling: predict next token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward pass through embedding + model
        with torch.cuda.amp.autocast():
            # Simple embedding
            embeddings = F.embedding(inputs, torch.randn(256, 256, device=device))
            output = model(embeddings)
            
            # Project to vocab for next token prediction
            logits = F.linear(output, torch.randn(256, 256, device=device))
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        step_time = time.time() - start_time
        
        losses.append(loss.item())
        times.append(step_time)
        
        if step % 3 == 0 or step == 9:
            print(f"Step {step+1:2d}: loss={loss.item():.4f}, time={step_time:.3f}s, "
                  f"grad_norm={get_grad_norm(model):.4f}")
    
    avg_loss = sum(losses) / len(losses)
    avg_time = sum(times) / len(times)
    
    print(f"{model_name} Results:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average time per step: {avg_time:.3f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss improvement: {losses[0] - losses[-1]:.4f}")
    
    return {
        'avg_loss': avg_loss,
        'avg_time': avg_time,
        'final_loss': losses[-1],
        'improvement': losses[0] - losses[-1],
        'losses': losses
    }

def get_grad_norm(model):
    """Get gradient norm for monitoring"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def create_fixed_model():
    """Create model with FixedGRU"""
    from mingru.minLM import minLM
    return minLM(
        num_tokens=256,
        dim=256,
        depth=8,
        ff_mult=2,
        expansion=1.5,
        use_nau=True,  # This will use FixedGRU now
        use_barriers=False
    )

def create_minrgu_model():
    """Create baseline minGRU model"""
    from mingru.minLM import minLM
    return minLM(
        num_tokens=256,
        dim=256,
        depth=8,
        ff_mult=2,
        expansion=1.5,
        use_nau=False  # Use standard minGRU
    )

def create_original_nau_model():
    """Create model with original (buggy) NAU_GRU"""
    # Temporarily modify import to use original
    import mingru.minLM as minLM_module
    
    # Save current import
    current_nau = minLM_module.NAU_GRU
    
    try:
        # Import original buggy version
        from mingru.nau_gru_cell import NAU_GRU as OriginalNAU
        minLM_module.NAU_GRU = OriginalNAU
        
        model = minLM_module.minLM(
            num_tokens=256,
            dim=256,
            depth=8,
            ff_mult=2,
            expansion=1.5,
            use_nau=True,
            use_barriers=True,
            barrier_min=-8,
            barrier_max=8
        )
        return model
    finally:
        # Restore
        minLM_module.NAU_GRU = current_nau

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU (will be slow)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    print("Creating synthetic training data...")
    data = create_synthetic_data(batch_size=4, seq_len=64, num_batches=10)
    
    results = {}
    
    # Test FixedGRU (new)
    try:
        results['FixedGRU'] = test_model_on_synthetic_data(
            "FixedGRU", create_fixed_model, data, device
        )
    except Exception as e:
        print(f"Error testing FixedGRU: {e}")
        results['FixedGRU'] = None
    
    # Test minGRU (baseline)
    try:
        results['minGRU'] = test_model_on_synthetic_data(
            "minGRU", create_minrgu_model, data, device
        )
    except Exception as e:
        print(f"Error testing minGRU: {e}")
        results['minGRU'] = None
    
    # Test original NAU_GRU (buggy)
    try:
        results['Original_NAU'] = test_model_on_synthetic_data(
            "Original NAU_GRU", create_original_nau_model, data, device
        )
    except Exception as e:
        print(f"Error testing Original NAU_GRU: {e}")
        results['Original_NAU'] = None
    
    # Summary
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    for name, result in results.items():
        if result is not None:
            print(f"{name:15s}: final_loss={result['final_loss']:.4f}, "
                  f"improvement={result['improvement']:.4f}, "
                  f"time={result['avg_time']:.3f}s/step")
        else:
            print(f"{name:15s}: FAILED")
    
    if results.get('FixedGRU') and results.get('Original_NAU'):
        fixed_loss = results['FixedGRU']['final_loss']
        orig_loss = results['Original_NAU']['final_loss']
        improvement = orig_loss - fixed_loss
        print(f"\nFixedGRU vs Original NAU_GRU:")
        print(f"  Loss improvement: {improvement:.4f}")
        print(f"  Relative improvement: {improvement/orig_loss*100:.1f}%")

if __name__ == "__main__":
    main()
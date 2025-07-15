#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from mingru.minLM import minLM

def load_model(checkpoint_path, device='cpu'):
    """Load model - copied from your generate.py"""
    if os.path.isdir(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        auto_config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        with open(auto_config_path, 'r') as f:
            config = json.load(f)
    
    model_params = {
        "num_tokens": config.get("num_tokens", 256),
        "dim": config.get("dim"),
        "depth": config.get("depth"),
        "ff_mult": config.get("ff_mult", 4.0),
        "expansion": config.get("expansion", 1.5),
        "enable_conv": config.get("enable_conv", False),
        "dropout": config.get("dropout", 0.0)
    }
    
    model = minLM(**model_params)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"INFO: Model from training step: {checkpoint.get('step', 'unknown')}")
    print(f"INFO: EMA loss at checkpoint: {checkpoint.get('ema_fitness', 'unknown'):.4f}")
    
    model = model.eval().to(device)
    print(f"INFO: Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")
    return model

def decode_token(token):
    """Decode single token - same as generate.py"""
    return chr(token) if 32 <= token <= 126 else f'\\x{token:02x}'

def decode_tokens(tokens):
    """Decode list of tokens - same as generate.py"""
    return "".join(map(decode_token, tokens))

def diffusion_generate(model, prompt, length=256, num_steps=20, device='cpu', verbose=True):
    """
    Core diffusion generation using parallel mode with step-by-step output.
    """
    print(f"\nStarting diffusion generation:")
    print(f"- Steps: {num_steps}")
    print(f"- Length: {length} tokens")
    print(f"- Device: {device}")
    print("-" * 80)
    
    # Initialize with random tokens
    x_t = torch.randint(0, 256, (1, length), device=device)
    
    # Show initial noise
    if verbose:
        initial_text = decode_tokens(x_t[0].cpu().tolist())
        print(f"Step 00/{num_steps} [NOISE]: '{initial_text[:80]}{'...' if len(initial_text) > 80 else ''}'")
    
    # Temperature schedule - start high, end low
    temperatures = torch.linspace(2.0, 0.1, num_steps)
    
    for step in range(num_steps):
        t = step / (num_steps - 1)  # 0 to 1 progress
        temp = temperatures[step].item()
        
        # Full sequence for parallel processing
        full_seq = torch.cat([prompt, x_t], dim=1)
        
        with torch.no_grad():
            # This uses the PARALLEL branch - same as training!
            logits, _ = model(full_seq, return_prev_hiddens=True)
            
            # Get logits only for generated portion
            gen_logits = logits[:, prompt.shape[1]:, :]
            
            # Apply temperature and get probabilities
            probs = F.softmax(gen_logits / temp, dim=-1)
            
            # Sample new tokens
            new_tokens = torch.multinomial(probs.view(-1, 256), 1).view(1, length)
            
            # Gradually trust model more (stochastic mixing)
            trust_model = t  # 0->1 over steps
            update_mask = torch.rand(length, device=device) < trust_model
            
            # Update tokens
            old_x_t = x_t.clone()
            x_t = torch.where(update_mask.unsqueeze(0), new_tokens, x_t)
            
            # Show current state
            if verbose:
                current_text = decode_tokens(x_t[0].cpu().tolist())
                changed = update_mask.sum().item()
                
                # Show what changed (optional - can be noisy)
                changes = (old_x_t != x_t).sum().item()
                
                print(f"Step {step+1:02d}/{num_steps} [T={temp:.2f}, changed={changed:3d}]: '{current_text[:80]}{'...' if len(current_text) > 80 else ''}'")
    
    print("-" * 80)
    return x_t

def main():
    parser = argparse.ArgumentParser(description="Diffusion generation for minGRU")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--prompt", type=str, default="The CPU processes instructions")
    parser.add_argument("--length", type=int, default=256, help="Generation length")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Don't show intermediate steps")
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"INFO: Using seed {args.seed}")
    
    # Setup device - EXACTLY like generate.py
    if args.cpu:
        device = 'cpu'
    elif args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"INFO: Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    # Encode prompt
    prompt_bytes = list(args.prompt.encode('utf-8'))
    prompt_tensor = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nPrompt ({len(prompt_bytes)} bytes): '{args.prompt}'")
    
    # Generate!
    start_time = time.time()
    generated = diffusion_generate(
        model, prompt_tensor, args.length, args.steps, 
        device, verbose=not args.quiet
    )
    gen_time = time.time() - start_time
    
    # Decode final result
    generated_bytes = generated[0].cpu().tolist()
    generated_text = decode_tokens(generated_bytes)
    
    print(f"\nFINAL RESULT:")
    print(f"'{generated_text}'")
    
    print(f"\nStats:")
    print(f"- Generation time: {gen_time:.2f}s ({args.steps} steps = {gen_time/args.steps:.2f}s/step)")
    print(f"- Tokens/sec: {args.length / gen_time:.2f}")
    
    # Save to file if needed
    print("\n[Run with --seed <num> to reproduce this generation]")

if __name__ == "__main__":
    main()
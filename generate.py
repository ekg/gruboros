import os
import argparse
import gzip
import random
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
import time
import sys
import mmap

# Set higher precision for float32 matrix multiplication
# This enables TensorFloat32 on supported GPUs
torch.set_float32_matmul_precision('high')

# Import the minLM model
from mingru.minLM import minLM

# Token decoding function
def decode_token(token):
    # The model works with raw bytes (0-255). We display printable ASCII.
    return chr(token) if 32 <= token <= 126 else f'\\x{token:02x}'

def decode_tokens(tokens):
    return "".join(map(decode_token, tokens))

# --- 1. UNIFIED AND EFFICIENT SAMPLING FUNCTION ---
def sample(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    A single, flexible sampling function supporting temperature, top-k, and top-p.
    All operations are performed in PyTorch on the target device.

    Args:
        logits: Raw model logits [batch_size, vocab_size].
        temperature (float): Controls randomness. 0 for greedy.
        top_k (int): If > 0, keep only top k tokens.
        top_p (float): If > 0.0, keep tokens with cumulative probability >= top_p.

    Returns:
        torch.Tensor: The sampled token indices.
    """
    if temperature == 0.0:
        # Greedy sampling
        return logits.argmax(dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        logits = filtered_logits

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Fix: Use the correct dimension and pattern for scatter
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        logits[indices_to_remove] = -float('Inf')
        
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def locally_typical_sampling(logits, temperature=1.0, typical_p=0.9, typical_mass=0.9, epsilon=1e-10):
    """
    Locally typical sampling: Sample from tokens whose information content is close to expected.
    
    Args:
        logits: Raw model logits [batch_size, vocab_size].
        temperature: Controls randomness. Lower = less random.
        typical_p: Mass of typical set to consider (similar to top_p).
        typical_mass: Alternative to typical_p - include tokens until we have this much mass.
        epsilon: Small value for numerical stability.
    
    Returns:
        torch.Tensor: The sampled token indices.
    """
    if temperature == 0.0:
        # Greedy sampling
        return logits.argmax(dim=-1, keepdim=True)
    
    # Apply temperature
    logits = logits / temperature
    
    # Get probabilities and log probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Calculate entropy (expected information content)
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
    
    # Calculate absolute deviation from entropy for each token
    neg_log_probs = -log_probs
    typicality = torch.abs(neg_log_probs - entropy)
    
    # Sort by typicality (ascending - most typical first)
    sorted_typicality, sorted_indices = torch.sort(typicality, dim=-1)
    sorted_probs = torch.gather(probs, -1, sorted_indices)
    
    # Calculate cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for typical set (include tokens until we have typical_p mass)
    mask = cumsum_probs < typical_p
    # Ensure we include at least one token
    mask[..., 0] = True
    
    # Zero out probabilities outside typical set
    sorted_probs[~mask] = 0.0
    
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    sample_idx = torch.multinomial(sorted_probs.squeeze(0), num_samples=1)
    
    # Map back to original indices
    original_idx = sorted_indices.squeeze(0)[sample_idx]
    
    return original_idx.unsqueeze(0)

def min_p_sampling(logits, temperature=1.0, min_p=0.1):
    """
    Min-P sampling: Include all tokens with probability >= min_p * max_prob
    
    Args:
        logits: Raw model logits [batch_size, vocab_size]
        temperature: Temperature for softmax
        min_p: Minimum probability threshold as fraction of max probability
    
    Returns:
        Sampled token indices
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    
    # Apply temperature and get probabilities
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Find maximum probability
    max_prob, _ = probs.max(dim=-1, keepdim=True)
    
    # Create threshold: min_p * max_prob
    threshold = min_p * max_prob
    
    # Create mask for tokens above threshold
    mask = probs >= threshold
    
    # Ensure at least one token is selected
    if not mask.any(dim=-1).all():
        # If no tokens meet threshold, select the maximum
        mask = probs == max_prob
    
    # Zero out probabilities below threshold
    filtered_probs = probs * mask.float()
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    return torch.multinomial(filtered_probs, num_samples=1)

def load_model(checkpoint_path, config_path=None, use_bf16=False, use_fp16=False, device=None):
    """
    Load a trained minLM model from checkpoint. (Largely unchanged, but still good)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.isdir(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_path) and os.path.islink(latest_path):
            real_ckpt_name = os.path.basename(os.readlink(latest_path))
            print(f"INFO: Using latest checkpoint: {real_ckpt_name}")
            checkpoint_path = latest_path
        elif os.path.exists(latest_path):
             print(f"INFO: Using checkpoint: {os.path.basename(latest_path)}")
             checkpoint_path = latest_path
        else:
            raise ValueError(f"No 'latest.pt' symlink or file found in directory: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("INFO: Using model configuration from checkpoint.")
    else:
        auto_config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.exists(auto_config_path):
            with open(auto_config_path, 'r') as f: config = json.load(f)
            print(f"INFO: Using model configuration from {auto_config_path}")
        else:
            raise ValueError(f"Could not find config.json in checkpoint directory: {os.path.dirname(checkpoint_path)}")
    
    # Make model loading robust to older configs that might be missing keys
    model_params = {
        "num_tokens": config.get("num_tokens", 256),
        "dim": config.get("dim"),
        "depth": config.get("depth"),
        "ff_mult": config.get("ff_mult", 4.0),
        "expansion": config.get("expansion", 1.5),
        "enable_conv": config.get("enable_conv", False),
        "dropout": config.get("dropout", 0.0)
    }
    
    print(f"INFO: Creating model with dimension={model_params['dim']}, depth={model_params['depth']}...")
    model = minLM(**model_params)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        print("INFO: Adapted weights from a compiled model checkpoint.")

    model.load_state_dict(state_dict)
    
    if 'step' in checkpoint: print(f"INFO: Model from training step: {checkpoint['step']}")
    if 'ema_fitness' in checkpoint: print(f"INFO: EMA loss at checkpoint: {checkpoint['ema_fitness']:.4f}")
    
    model = model.eval()
    
    if device == 'cuda':
        if use_bf16 and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
        elif use_fp16:
            model = model.to(torch.float16)
    
    model = model.to(device)
    print(f"INFO: Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")
    
    return model

# --- 2. REFACTORED, EFFICIENT, AND STREAMING GENERATION FUNCTION ---
def generate(
    model,
    prompt: torch.Tensor,
    generation_length: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    typical_p: float = 0.0,
    min_p: float = 0.0,  # Add min-p parameter
    device: str = 'cuda',
    stream: bool = True
):
    """
    Efficient, stateful, and streaming text generation for minLM.

    Args:
        model: The minLM model.
        prompt: Starting prompt tensor [1, seq_len].
        generation_length: Number of tokens to generate.
        temperature, top_k, top_p: Standard sampling parameters.
        typical_p: If > 0, use locally typical sampling instead of top-k/top-p.
        min_p: If > 0, use min-p sampling instead of top-k/top-p.
        device: Device to run on.
        stream: Whether to print tokens as they are generated.
    """
    model.eval()
    prompt = prompt.to(device)
    
    generated_tokens = []
    start_time = time.time()

    with torch.no_grad():
        # --- 1. EFFICIENT PROMPT PROCESSING ---
        # Process the entire prompt in one parallel forward pass to get the initial hidden state.
        # This is MUCH faster than looping token-by-token in Python.
        print("INFO: Processing prompt...")
        logits, hidden_states = model(prompt, return_prev_hiddens=True)

        # The next token is sampled from the logits of the last token in the prompt
        last_logits = logits[:, -1, :] # Shape: [batch_size, vocab_size]
        
        # Choose sampling method
        if typical_p > 0:
            prev_token = locally_typical_sampling(last_logits, temperature, typical_p)
        elif min_p > 0:
            prev_token = min_p_sampling(last_logits, temperature, min_p)
        else:
            prev_token = sample(last_logits, temperature, top_k, top_p)
        
        if stream:
            # Print the prompt itself for context
            sys.stdout.write("PROMPT: '" + decode_tokens(prompt.tolist()[0]) + "'")
            sys.stdout.write("\nGENERATED: '")
            token_val = prev_token.item()
            generated_tokens.append(token_val)
            sys.stdout.write(decode_token(token_val))
            sys.stdout.flush()

        # --- 2. AUTOREGRESSIVE GENERATION LOOP ---
        # Now, we generate token-by-token, which is inherently sequential.
        for step in range(1, generation_length):
            # Pass the single previous token and the current hidden state
            logits, hidden_states = model(
                prev_token,
                return_prev_hiddens=True,
                prev_hiddens=hidden_states
            )
            
            # Sample the next token
            if typical_p > 0:
                prev_token = locally_typical_sampling(logits[:, -1, :], temperature, typical_p)
            elif min_p > 0:
                prev_token = min_p_sampling(logits[:, -1, :], temperature, min_p)
            else:
                prev_token = sample(logits[:, -1, :], temperature, top_k, top_p)
            
            token_val = prev_token.item()
            generated_tokens.append(token_val)

            if stream:
                sys.stdout.write(decode_token(token_val))
                sys.stdout.flush()
    
    # Final cleanup for streaming output
    if stream:
        sys.stdout.write("'\n")
        total_time = time.time() - start_time
        tokens_per_sec = generation_length / total_time if total_time > 0 else 0
        print(f"\nINFO: Generation complete! ({tokens_per_sec:.2f} tokens/sec)")

    return torch.tensor(generated_tokens, device=device).unsqueeze(0)

def load_primer_text(primer_file=None, primer_length=None, primer_text=None, explicit_length=False):
    if primer_text:
        # Convert text to bytes using UTF-8 encoding
        byte_data = primer_text.encode('utf-8')
        tokens = list(byte_data)  # This gives us values 0-255
        if explicit_length and primer_length and len(tokens) > primer_length:
            print(f"WARNING: Primer text truncated from {len(tokens)} to {primer_length} tokens.")
            tokens = tokens[:primer_length]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]
    
    elif primer_file:
        with open(primer_file, 'r', encoding='utf-8') as f: text = f.read()
        # Convert text to bytes using UTF-8 encoding
        byte_data = text.encode('utf-8')
        tokens = list(byte_data)  # This gives us values 0-255
        if primer_length and len(tokens) > primer_length:
            print(f"WARNING: Primer file truncated from {len(tokens)} to {primer_length} tokens.")
            tokens = tokens[:primer_length]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]
    else:
        # Default prompt
        byte_data = "The ".encode('utf-8')
        tokens = list(byte_data)
        return torch.tensor(tokens, dtype=torch.long)[None, ...]

def parse_size_with_suffix(size_str):
    if not isinstance(size_str, str): return size_str
    pattern = r'^(\d+(?:\.\d+)?)([kmg])?$'
    match = re.match(pattern, size_str.lower())
    if not match: return float(size_str)
    value, suffix = match.groups()
    value = float(value)
    if suffix == 'k': return int(value * 1024)
    elif suffix == 'm': return int(value * 1024 * 1024)
    elif suffix == 'g': return int(value * 1024 * 1024 * 1024)
    return int(value)

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained minLM model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint or directory.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', etc.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--use-f32", dest="use_bf16", action="store_false", default=True, help="Use FP32 instead of BF16.")
    parser.add_argument("--use-fp16", action="store_true", default=False, help="Use FP16 instead of BF16/FP32.")
    
    parser.add_argument("--generation_length", type=str, default="512", help="Tokens to generate (e.g., 512, 2k).")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature (0=greedy).")
    
    # --- Sampling method controls ---
    parser.add_argument("--typical", action="store_true", 
                        help="Use locally typical sampling instead of top-p/top-k.")
    parser.add_argument("--typical_p", type=float, default=0.9,
                        help="Typical sampling threshold (0.8-0.95 recommended). Only used with --typical.")
    
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Min-P sampling threshold (0.02-0.1 recommended). "
                             "Include tokens with prob >= min_p * max_prob.")
    
    # --- Existing sampling controls (only used if --typical is not set) ---
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability. (e.g., 0.9). Set to 0 to disable.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling. (e.g., 40). Set to 0 to disable. If both top-p and top-k are non-zero, top-k is applied first.")
    
    parser.add_argument("--primer_file", type=str, default=None, help="File with primer text.")
    parser.add_argument("--primer_text", type=str, default=None, help="Direct primer text.")
    parser.add_argument("--primer_length", type=str, default="1k", help="Max primer length (e.g., 256, 1k).")
    parser.add_argument("--force_primer_length", action="store_true", help="Force truncation of --primer_text.")
    
    parser.add_argument("--output_file", type=str, default=None, help="File to save the full generated text.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", default=True, help="Disable streaming output.")
    
    args = parser.parse_args()
    
    if args.cpu: device = 'cpu'
    elif args.device == "auto": device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: device = args.device
    print(f"INFO: Using device: {device}")
    
    model = load_model(
        checkpoint_path=args.model, use_bf16=args.use_bf16, use_fp16=args.use_fp16, device=device
    )
    
    generation_length = int(parse_size_with_suffix(args.generation_length))
    primer_length = int(parse_size_with_suffix(args.primer_length)) if args.primer_length else None
    
    prompt = load_primer_text(
        args.primer_file, primer_length, args.primer_text, args.force_primer_length
    ).long().to(device)
    
    # Clear logging of sampling method
    if args.typical:
        sampling_params = f"Typical sampling with p={args.typical_p}, T={args.temperature}"
        # When using typical sampling, set other methods to 0
        top_p_value = 0.0
        top_k_value = 0
        typical_p_value = args.typical_p
        min_p_value = 0.0
    elif args.min_p > 0:
        sampling_params = f"Min-P sampling with p={args.min_p}, T={args.temperature}"
        # When using min-p sampling, set other methods to 0
        top_p_value = 0.0
        top_k_value = 0
        typical_p_value = 0.0
        min_p_value = args.min_p
    else:
        sampling_params = f"T={args.temperature}"
        if args.top_p > 0:
            sampling_params += f", top_p={args.top_p}"
        if args.top_k > 0:
            sampling_params += f", top_k={args.top_k}"
        top_p_value = args.top_p
        top_k_value = args.top_k
        typical_p_value = 0.0
        min_p_value = 0.0
    
    print(f"\nINFO: Generating {generation_length} tokens with sampling: {sampling_params}")
    
    generated_tensor = generate(
        model, prompt, generation_length,
        args.temperature, top_k_value, top_p_value,
        typical_p_value, min_p_value,  # Pass typical_p and min_p to generate
        device, args.stream
    )
    
    if args.output_file:
        full_text = decode_tokens(prompt.tolist()[0]) + decode_tokens(generated_tensor.tolist()[0])
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"INFO: Full output saved to {args.output_file}")

if __name__ == "__main__":
    main()

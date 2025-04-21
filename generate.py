import os
import argparse
import gzip
import random
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from torch import Tensor
import time
import mmap

# Set higher precision for float32 matrix multiplication
# This enables TensorFloat32 on supported GPUs
torch.set_float32_matmul_precision('high')

# Import the minLM model
from mingru.minLM import minLM

# Token decoding functions
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# Sampling helpers
def improved_top_k_sampling(logits, temperature=1.0, top_k=40):
    """
    Stable top-k sampling implementation using direct probability sampling
    
    Args:
        logits: Raw logits from model output [batch_size, vocab_size]
        temperature: Controls randomness (higher = more random)
        top_k: Number of highest probability tokens to consider
    
    Returns:
        Sampled token indices
    """
    # Apply temperature scaling
    if temperature > 0:
        scaled_logits = logits / temperature
    else:
        # If temperature is 0, use greedy sampling (just return argmax)
        return logits.argmax(dim=-1, keepdim=True)
        
    batch_size, vocab_size = scaled_logits.shape
    device = logits.device
    
    # For each item in the batch
    selected_tokens = []
    
    for i in range(batch_size):
        # Get logits for this batch item
        item_logits = scaled_logits[i].detach().cpu().numpy()
        
        # Find indices of top-k values
        top_k_indices = np.argpartition(item_logits, -top_k)[-top_k:]
        
        # Get the corresponding logit values
        top_k_logits = item_logits[top_k_indices]
        
        # Apply softmax to just these k values to get probabilities
        top_k_probs = F.softmax(torch.tensor(top_k_logits), dim=0).numpy()
        
        # Sample using cumulative probability (inverse CDF sampling)
        cumulative_probs = np.cumsum(top_k_probs)
        random_value = np.random.random()
        selected_idx = 0
        
        for j, cumprob in enumerate(cumulative_probs):
            if random_value < cumprob:
                selected_idx = j
                break
        
        # Map back to original token index
        selected_token = top_k_indices[selected_idx]
        selected_tokens.append(selected_token)
    
    # Convert to tensor and reshape appropriately
    return torch.tensor(selected_tokens, device=device).unsqueeze(-1)

def top_k(logits, k=40):
    """Top-k sampling to limit generated tokens to top k candidates
    
    Args:
        logits: Raw logits from model output
        k: Number of top tokens to keep (default: 40)
    
    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Convert logits to probabilities with softmax
    probs = F.softmax(logits, dim=-1)
    
    # Ensure k is at least 1 and not larger than vocab size
    k = max(1, min(k, logits.shape[-1]))
    
    # Get the top k tokens
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    
    # Create a mask with -inf for tokens not in top k
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, top_indices, torch.log(top_probs + 1e-10))
    
    return filtered_logits

def top_p(logits, thres=0.9):
    """Nucleus (top-p) sampling
    
    Args:
        logits: Raw logits from model output
        thres: Value between 0 and 1 representing cumulative probability threshold
    
    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > thres
    
    # Shift the indices to the right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Create a mask for tokens to keep
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Set -inf where mask is True
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    
    return filtered_logits

def load_model(checkpoint_path, config_path=None, use_bf16=False, use_fp16=False, device=None):
    """
    Load a trained minLM model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the model config file (optional)
        use_bf16: Whether to load model in BF16 precision (default: False)
        use_fp16: Whether to load model in FP16 precision (default: False)
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded model
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Support directory paths - use latest.pt if a directory is provided
    if os.path.isdir(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
            print(f"Using latest checkpoint: {latest_path}")
        else:
            raise ValueError(f"No 'latest.pt' file found in directory: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)  # First load to CPU to avoid OOM
    
    # Try to get config from checkpoint first
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("Using model configuration from checkpoint")
    elif config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Using model configuration from {config_path}")
    else:
        # Look for config.json in the same directory as the checkpoint
        auto_config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.exists(auto_config_path):
            with open(auto_config_path, 'r') as f:
                config = json.load(f)
            print(f"Using model configuration from {auto_config_path}")
        else:
            # For backward compatibility, try model_config.json
            legacy_config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
            if os.path.exists(legacy_config_path):
                with open(legacy_config_path, 'r') as f:
                    config = json.load(f)
                print(f"Using model configuration from {legacy_config_path} (legacy)")
            else:
                # Warn the user that no config was found
                print(f"WARNING: No config.json found in {os.path.dirname(checkpoint_path)}")
                print("Model configuration is required for correct generation.")
                if not config_path:
                    raise ValueError(
                        "No model configuration found. Please provide a config file path with --config_path."
                    )
                else:
                    raise ValueError(f"Provided config path {config_path} does not exist.")
    
    print(f"Creating model with dimension={config['dim']}, depth={config['depth']}...")
    
    # Create model with the correct configuration
    model = minLM(
        num_tokens=config["num_tokens"],
        dim=config["dim"],
        depth=config["depth"],
        ff_mult=config["ff_mult"],
        expansion=config.get("expansion", 1.5),
        conv_kernel_size=config.get("conv_kernel_size", 3),
        use_lstm=config.get("use_lstm", False),
        enable_conv=config.get("enable_conv", False),
        dropout=config.get("dropout", 0.0)
    )
    
    # Load model weights - handling different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Our custom checkpoint format
        # First check if this is a compiled model (has '_orig_mod.' prefix)
        if any(key.startswith('_orig_mod.') for key in checkpoint['model_state_dict']):
            print("Detected compiled model checkpoint (_orig_mod. prefix)")
            # Remove the '_orig_mod.' prefix from all keys
            fixed_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('_orig_mod.'):
                    fixed_state_dict[key[10:]] = value  # Remove '_orig_mod.' prefix
                else:
                    fixed_state_dict[key] = value
            model.load_state_dict(fixed_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        # Standard Lightning checkpoint format
        pl_state_dict = checkpoint['state_dict']
            
        # The model in LightningMinLM is stored under 'model.' prefix
        model_state_dict = {}
        for key, value in pl_state_dict.items():
            # Remove the 'model.' prefix from keys
            if key.startswith('model.'):
                model_state_dict[key[6:]] = value
            
        model.load_state_dict(model_state_dict)
    elif 'model' in checkpoint and 'state_dict' in checkpoint['model']:
        # Another possible Lightning format
        model.load_state_dict(checkpoint['model']['state_dict'])
    else:
        # Raw state dict - try direct loading
        try:
            model.load_state_dict(checkpoint)
        except (RuntimeError, KeyError) as e:
            # Handle DeepSpeed formats by checking for different module prefixes
            print(f"Direct loading failed, trying to match keys: {str(e)}")
            if isinstance(checkpoint, dict):
                # Try to adapt keys if they don't match directly
                ds_state_dict = {}
                # Check for module prefixes used by DeepSpeed
                for key, value in checkpoint.items():
                    if key.startswith('module.model.'):
                        # DeepSpeed might add 'module.' prefix
                        ds_state_dict[key[13:]] = value  # Remove 'module.model.'
                    elif key.startswith('model.'):
                        ds_state_dict[key[6:]] = value  # Remove 'model.'
                    elif key.startswith('_orig_mod.'):
                        # Handle compiled model saved with torch.compile
                        ds_state_dict[key[10:]] = value  # Remove '_orig_mod.' prefix
                    else:
                        ds_state_dict[key] = value  # Keep as is
                    
                # Try loading with adapted keys
                model.load_state_dict(ds_state_dict)
                print("Successfully loaded model with adapted keys")
    
    # Print additional info from checkpoint
    if 'step' in checkpoint:
        print(f"Model checkpoint from training step: {checkpoint['step']}")
    if 'train_loss' in checkpoint:
        print(f"Training loss at checkpoint: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"Validation loss at checkpoint: {checkpoint['val_loss']:.4f}")
    elif 'val_loss' in checkpoint:
        print(f"Validation loss at checkpoint: None")
    
    # Set model to evaluation mode
    model = model.eval()
    
    # Apply precision conversion
    if device == 'cuda':
        if use_bf16 and torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            print("Model converted to BF16 precision")
        elif use_fp16 and torch.cuda.is_available():
            model = model.to(torch.float16)
            print("Model converted to FP16 precision")
    
    # Move model to device
    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    
    return model

def chunked_generation(
    model,
    prompt: torch.Tensor,
    generation_length: int,
    chunk_length: int,
    temperature: float = 1.0,
    filter_thres: int = 40,
    sampling_method: str = 'top_k',  # 'top_k' or 'top_p'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    callback=None,
    top_p_tokens: int = 40
):
    """
    Generate text in chunks using RNN hidden states.
    
    Args:
        model: The minLM model
        prompt: Starting prompt tensor
        generation_length: Total length to generate
        chunk_length: Length to process in each forward pass
        temperature: Temperature for sampling (higher = more random)
        filter_thres: Threshold for top-k filtering (higher = more diverse)
        device: Device to run generation on
        callback: Optional callback function to report progress
    
    Returns:
        Generated tokens
    """
    # Move prompt to device
    prompt = prompt.to(device)
    
    # Start with the prompt
    out = prompt.clone()
    
    # Track number of tokens to generate
    remaining_tokens = generation_length
    
    # Start with no hidden state
    prev_hiddens = None
    
    # Process the prompt efficiently to get initial hidden state
    with torch.no_grad():
        # Process the entire prompt at once to get the initial hidden state
        _, prev_hiddens = model(out, return_prev_hiddens=True)
    
    # Timing variables
    start_time = time.time()
    tokens_generated = 0
    
    # Efficient memory management
    torch.cuda.empty_cache()
    
    # Generate tokens in chunks
    with torch.no_grad():
        while remaining_tokens > 0:
            # Generate tokens efficiently in batches
            batch_size = min(chunk_length, remaining_tokens)
            
            # Generate tokens one at a time, efficiently maintaining RNN state
            for _ in range(batch_size):
                # Get logits and updated hidden state
                logits, next_prev_hiddens = model(
                    out[:, -1:],  # Only need the last token with RNN state
                    return_prev_hiddens=True, 
                    prev_hiddens=prev_hiddens
                )
                
                # Get logits for the generated token
                logits = logits[:, -1]
                
                # Update hidden state for next iteration
                prev_hiddens = next_prev_hiddens
                
                # Apply sampling based on method chosen
                if sampling_method == 'top_p':
                    # First filter with top-p
                    filtered_logits = top_p(logits, thres=filter_thres)
                    # Then apply sampling
                    sample = improved_top_k_sampling(filtered_logits, temperature, top_k=top_p_tokens)
                else:
                    # Direct top-k sampling
                    sample = improved_top_k_sampling(logits, temperature, top_k=filter_thres)
                
                # Append the new token to the output
                out = torch.cat((out, sample), dim=-1)
                
                tokens_generated += 1
                
                # Update progress regularly
                if tokens_generated % 5 == 0 and callback:
                    progress = tokens_generated / generation_length
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
                    callback(progress, tokens_per_sec)
            
            # Update remaining tokens
            remaining_tokens -= batch_size
            
            # Calculate and show progress for the whole chunk
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
            
            # Call progress callback if provided
            if callback:
                progress = (generation_length - remaining_tokens) / generation_length
                callback(progress, tokens_per_sec)
            
            # Periodically clear unused memory
            if tokens_generated % 100 == 0:
                torch.cuda.empty_cache()
    
    # Return only the newly generated tokens (excluding the prompt)
    return out[..., prompt.shape[-1]:]

def load_primer_text(primer_file=None, primer_length=None, val_dataset=None):
    """
    Load primer text either from a file, or randomly from validation dataset
    """
    if primer_file:
        # Load from file
        with open(primer_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Convert to tensor of byte values
        tokens = [ord(c) for c in text]
        if primer_length:
            tokens = tokens[:primer_length]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension
    
    elif val_dataset:
        # Random sample from validation set
        inp = random.choice(val_dataset)
        if primer_length:
            inp = inp[:primer_length]
        # Ensure this is a Long tensor
        return inp.long()[None, ...]  # Add batch dimension and convert to long
    
    else:
        # Default to a simple prompt if no primer is provided
        text = "The "
        tokens = [ord(c) for c in text]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension

def parse_size_with_suffix(size_str):
    """
    Parse a string with optional k, m, g suffix into a number.
    Examples:
      "1k" -> 1024
      "100k" -> 102400 (100*1024)
      "2m" -> 2097152 (2*1024*1024)
      "3g" -> 3221225472 (3*1024*1024*1024)
      "42" -> 42 (no suffix, unchanged)
    """
    if not isinstance(size_str, str):
        return size_str
        
    pattern = r'^(\d+(?:\.\d+)?)([kmg])?$'
    match = re.match(pattern, size_str.lower())
    if not match:
        try:
            return float(size_str)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")
            
    value, suffix = match.groups()
    value = float(value)
    
    if suffix == 'k':
        return value * 1024
    elif suffix == 'm':
        return value * 1024 * 1024
    elif suffix == 'g':
        return value * 1024 * 1024 * 1024
    else:
        return value

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained minLM model")
    
    # Model and data parameters
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint or directory")
    parser.add_argument("--config_path", type=str, default=None, 
                       help="Path to model config file (required if no config.json found with checkpoint)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: 'cpu', 'cuda', 'cuda:0', etc. (default: 'auto')")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution regardless of CUDA availability")
    parser.add_argument("--use-f32", dest="use_bf16", action="store_false", default=True,
                        help="Use FP32 precision instead of BF16 (default: BF16)")
    parser.add_argument("--use-fp16", action="store_true", default=False,
                        help="Use FP16 precision instead of BF16/FP32 (default: False)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=40, help="Number of tokens to consider with top-k sampling (default: 40)")
    parser.add_argument("--use_top_p", action="store_true", help="Use nucleus (top-p) sampling instead of top-k")
    parser.add_argument("--top_p_tokens", type=int, default=40, help="Number of tokens to consider with top-p sampling (default: 40)")
    parser.add_argument("--chunk_length", type=str, default="64", help="Process sequence in chunks of this length (default: 64). Can use k/m/g suffix.")
    parser.add_argument("--generation_length", type=str, default="512", help="Total number of tokens to generate (default: 512). Can use k/m/g suffix.")
    
    # Input parameters
    parser.add_argument("--primer_file", type=str, default=None, help="File containing primer text (optional)")
    parser.add_argument("--primer_text", type=str, default=None, help="Direct text to use as primer")
    parser.add_argument("--primer_length", type=str, default="128", help="Length of primer sequence (default: 128). Can use k/m/g suffix.")
    parser.add_argument("--random_primer", action="store_true", help="Use a random primer from validation set")
    parser.add_argument("--data", type=str, default="./data/enwik8.gz", help="Path to data file for random primer (default: ./data/enwik8.gz)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write generated text (optional)")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient generation (slower but uses less VRAM)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        device = 'cpu'
        print("Forcing CPU execution as requested")
    elif args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(
        checkpoint_path=args.model, 
        config_path=args.config_path, 
        use_bf16=args.use_bf16,
        use_fp16=args.use_fp16,
        device=device
    )
    
    # Helper function to detect if a file is gzipped
    def is_gzip_file(filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    
    # Prepare validation dataset if using random primer
    val_dataset = None
    if args.random_primer:
        data_path = args.data
        print(f"Loading validation dataset for random primer from {data_path}...")
        
        if is_gzip_file(data_path):
            print("Detected gzip format, loading into memory...")
            with gzip.open(data_path) as file:
                data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
                _, np_valid = np.split(data, [int(90e6)])
                data_val = torch.from_numpy(np_valid)
        else:
            print("Detected raw format, using memory mapping...")
            # Get file size
            file_size = os.path.getsize(data_path)
            # Map the file into memory
            with open(data_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                # Create a numpy array using the memory map
                data = np.frombuffer(mm, dtype=np.uint8, count=min(int(95e6), file_size))
                # Get validation data
                train_size = min(int(90e6), len(data))
                np_valid = data[train_size:min(int(95e6), len(data))]
                data_val = torch.from_numpy(np_valid)
        
        # Create a simple dataset just for primer selection
        from torch.utils.data import Dataset
        
        class TextSamplerDataset(Dataset):
            def __init__(self, data, seq_len):
                super().__init__()
                # Ensure data is a Long tensor
                self.data = data.long() if data.dtype != torch.long else data
                self.seq_len = seq_len
            
            def __len__(self):
                return self.data.size(0) - self.seq_len
                
            def __getitem__(self, index):
                return self.data[index:index + self.seq_len + 1].long()
        
        val_dataset = TextSamplerDataset(data_val, args.primer_length)
    
    # Parse numerical arguments with potential suffixes
    chunk_length = int(parse_size_with_suffix(args.chunk_length))
    generation_length = int(parse_size_with_suffix(args.generation_length))
    primer_length = int(parse_size_with_suffix(args.primer_length)) if args.primer_length else None

    # If primer_text is provided directly, use that
    if args.primer_text:
        tokens = [ord(c) for c in args.primer_text]
        if primer_length:
            tokens = tokens[:primer_length]
        prompt = torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension
    else:
        # Otherwise get primer text from file or validation dataset
        prompt = load_primer_text(args.primer_file, primer_length, val_dataset)
    
    # Ensure prompt is a Long tensor before sending to device
    prompt = prompt.long().to(device)
    
    # Display the primer text
    primer_text = decode_tokens(prompt[0])
    print(f"\nPrimer text ({len(primer_text)} chars):\n{primer_text}")
    
    # Progress callback
    def progress_callback(progress, tokens_per_sec):
        percent_done = progress * 100
        print(f"Progress: {percent_done:.1f}% | Speed: {tokens_per_sec:.2f} tokens/sec", end="\r")
    
    print(f"\nGenerating {generation_length} tokens in chunks of {chunk_length}...")
    if args.use_top_p:
        print(f"Temperature: {args.temperature}, Nucleus (top-p) sampling with threshold: 0.9, considering top {args.top_p_tokens} tokens")
    else:
        print(f"Temperature: {args.temperature}, Top-k sampling with k={args.top_k} tokens")
    
    # Memory optimization
    if args.memory_efficient:
        print("Using memory-efficient generation mode")
        # Clear CUDA cache before generation
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    # Generate text
    start_time = time.time()
    with torch.no_grad():
        generated = chunked_generation(
            model,
            prompt,
            generation_length,
            chunk_length,
            args.temperature,
            args.top_k,
            'top_p' if args.use_top_p else 'top_k',
            device,
            progress_callback,
            top_p_tokens=args.top_p_tokens
        )
        
    # Clear cache after generation
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    total_tokens = generation_length
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Decode the generated text
    generated_text = decode_tokens(generated[0])
    
    # Display the generated text
    print(f"\n\nGeneration complete! ({tokens_per_sec:.2f} tokens/sec)")
    print(f"Generated text ({len(generated_text)} chars):")
    print(generated_text)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"PRIMER:\n{primer_text}\n\nGENERATED:\n{generated_text}")
        print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()

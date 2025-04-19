import os, random, numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import time
import argparse
import mmap
import re
import math
import json
import datetime
import sys
from tqdm import tqdm
from schedulefree import AdamWScheduleFree

# Set higher precision for float32 matrix multiplication 
# This enables TensorFloat32 on supported GPUs
torch.set_float32_matmul_precision('high')

# Import the minLM model
from mingru.minLM import minLM

# 1) Deterministic seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def round_to_multiple(n, multiple=64):
    """Round a number to the nearest multiple of a given value"""
    return multiple * round(n / multiple)

def solve_for_dimension(target_params, depth, vocab_size=256, ff_mult=4, expansion=1.5):
    """Solve for the dimension that gives the target parameter count"""
    factor = 4 * expansion + 2 * ff_mult
    
    # Quadratic equation: a*dim^2 + b*dim - target_params = 0
    a = depth * factor
    b = 2 * vocab_size
    c = -target_params
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No solution exists for the given target parameter count")
    
    dim = (-b + math.sqrt(discriminant)) / (2*a)
    return round_to_multiple(dim)

def solve_for_depth(target_params, dim, vocab_size=256, ff_mult=4, expansion=1.5):
    """Solve for the depth that gives the target parameter count"""
    embed_params = 2 * dim * vocab_size
    factor = 4 * expansion + 2 * ff_mult
    layer_params = dim * dim * factor
    
    depth = (target_params - embed_params) / layer_params
    return max(1, round(depth))

def calculate_model_size(config):
    """Calculate the number of parameters in the model"""
    dim = config["dim"]
    depth = config["depth"]
    vocab_size = config["num_tokens"]
    ff_mult = config["ff_mult"]
    expansion = config["expansion"]
    
    # Embedding and output layers
    embedding_params = dim * vocab_size
    output_params = dim * vocab_size
    
    # Each layer has:
    # - minGRU: 2*dim*dim*expansion + dim*dim (if expansion != 1)
    # - FF: dim*dim*ff_mult + dim*dim*ff_mult
    layer_params = dim * dim * (2 * expansion + 2 * ff_mult)
    
    # Total parameters
    total_params = embedding_params + output_params + depth * layer_params
    
    return int(total_params)

def get_parameter_count_str(config):
    """Get a human-readable string of parameter count"""
    params = calculate_model_size(config)
    
    if params >= 1_000_000_000:
        return f"{params/1_000_000_000:.1f}B"
    elif params >= 1_000_000:
        return f"{params/1_000_000:.1f}M"
    elif params >= 1_000:
        return f"{params/1_000:.1f}K"
    else:
        return f"{params}"

class ModuloSplitDataset(Dataset):
    """Dataset wrapper that allows modulo-based splitting"""
    def __init__(self, dataset, modulo, remainder, seed=42):
        self.dataset = dataset
        self.modulo = modulo
        self.remainder = remainder
        self.seed = seed
        
        # Generate indices that satisfy the modulo condition
        # This is much more efficient than checking each index at runtime
        self.indices = [i for i in range(len(dataset)) if i % modulo == remainder]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map the requested index to the corresponding filtered index
        return self.dataset[self.indices[idx]]

class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for efficient random access"""
    def __init__(self, filepath, seq_len, samples_per_epoch=10000):
        super().__init__()
        self.filepath = filepath
        self.seq_len = seq_len
        
        # Get file size once
        self.file_size = os.path.getsize(filepath)
        
        # Calculate valid end position for random sampling
        self.valid_end = max(0, self.file_size - seq_len - 1)
        
        # Define a fixed number of samples per epoch instead of file size
        # This prevents excessive data loading
        self.samples_per_epoch = samples_per_epoch
        
        # Create a shared memory map - single instance used across all workers
        self.file = open(self.filepath, 'rb')  # Read-only mode is sufficient
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Use the index deterministically to ensure same data across workers
        # But limit to a reasonable number of positions
        g = torch.Generator().manual_seed(SEED + idx)
        start_pos = torch.randint(0, self.valid_end, (1,), generator=g).item()
        
        # Get a slice from the memory map without reading the whole file
        # Thread-safe access
        with torch.no_grad():
            # Use seek/read with proper locking
            self.mm.seek(start_pos)
            data = self.mm.read(self.seq_len + 1)  # +1 for the target
            
            # Create a copy of the data in a writable buffer before making a tensor
            # This prevents the PyTorch warning about non-writable buffers
            data_copy = bytearray(data)
            
            # Create tensor from the copy (using list conversion to ensure it's writable)
            tensor = torch.tensor(list(data_copy), dtype=torch.long)
            
            # Handle edge case if we didn't get enough data
            if tensor.size(0) < self.seq_len + 1:
                # Pad with zeros if needed
                padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
                tensor = torch.cat([tensor, padding])
        
        return tensor
    
    def __del__(self):
        # Clean up resources
        try:
            if hasattr(self, 'mm') and self.mm is not None:
                self.mm.close()
            if hasattr(self, 'file') and self.file is not None:
                self.file.close()
        except:
            pass  # Avoid errors during interpreter shutdown

def get_model(model_config):
    """Create a minLM model with the given configuration"""
    model = minLM(
        num_tokens=model_config.get("num_tokens", 256),
        dim=model_config.get("dim", 512),
        depth=model_config.get("depth", 6),
        ff_mult=model_config.get("ff_mult", 4),
        expansion=model_config.get("expansion", 1.5),
        conv_kernel_size=model_config.get("conv_kernel_size", 3),
        use_lstm=model_config.get("use_lstm", False),
        enable_conv=model_config.get("enable_conv", False),
        dropout=model_config.get("dropout", 0.0)
    )
    
    # Use torch.compile to accelerate the model if available
    if hasattr(torch, 'compile') and callable(torch.compile):
        model = torch.compile(model)
        print("Model compiled with torch.compile")
    
    return model

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

def get_args():
    parser = argparse.ArgumentParser(
        description='DeepSpeed Tensor Parallel Training for minLM'
    )
    
    # Distributed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallel size')
    
    # Training arguments
    parser.add_argument('--train_steps', type=str, default="100",
                        help='number of training steps (default: 100)')
    parser.add_argument('--port', type=int, default=29500,
                        help='port for distributed communication (default: 29500)')
    parser.add_argument('--data', type=str, required=True,
                        help='path to training data file')
    parser.add_argument('--val_mod', type=int, default=10,
                        help='modulo for validation set (default: 10, meaning 1/10th for validation)')
    parser.add_argument('--output', type=str, default=None,
                        help='directory to save checkpoints (default: auto-generated)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save checkpoint every N steps (default: 100)')
    parser.add_argument('--validate_every', type=int, default=50,
                        help='validate every N steps (default: 50)')
    parser.add_argument('--force_lr', action='store_true',
                        help='force use command line learning rate when resuming')
    
    # Model configuration
    parser.add_argument('--dim', type=str, default="512", 
                        help='model hidden dimension (default: 512)')
    parser.add_argument('--depth', type=int, default=6,
                        help='number of transformer layers (default: 6)')
    parser.add_argument('--seq_len', type=str, default="128",
                        help='sequence length for training (default: 128)')
    parser.add_argument('--params', type=str, default=None,
                        help='target parameter count (e.g., 15m for 15M params)')
    parser.add_argument('--ff_mult', type=float, default=4.0,
                        help='feedforward multiplier (default: 4.0)')
    parser.add_argument('--expansion', type=float, default=1.5,
                        help='expansion factor for minGRU (default: 1.5)')
    
    # Optimizer configuration
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--batch_size', type=str, default="4",
                        help='batch size per GPU (default: 4)')
    parser.add_argument('--no-schedulefree', dest='schedulefree', action='store_false', default=True,
                        help='disable ScheduleFree optimizer (default: enabled)')
    parser.add_argument('--sf_beta', type=float, default=0.9,
                        help='ScheduleFree beta parameter (default: 0.9)')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set custom port for distributed communication
    if args.port != 29500:
        os.environ['MASTER_PORT'] = str(args.port)
    
    # Print memory usage monitoring message
    if args.local_rank == 0:
        print("Memory-efficient dataset initialized. Only loading necessary data.")
    
    # Parse numeric arguments with potential suffixes
    train_steps = int(parse_size_with_suffix(args.train_steps))
    seq_len = int(parse_size_with_suffix(args.seq_len))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    
    # Check for resuming from checkpoint
    resuming = args.resume is not None
    resume_step = 0
    loaded_config = None
    
    # Create output directory
    if args.local_rank == 0:
        if resuming:
            # Extract checkpoint directory from checkpoint path
            if os.path.isdir(args.resume):
                # If resume path is already a directory
                checkpoint_dir = args.resume
            else:
                # If resume path is a file, use its parent directory
                checkpoint_dir = os.path.dirname(args.resume)
            
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"Using checkpoint directory: {checkpoint_dir}")
            
            # Check if directory exists
            if not os.path.exists(checkpoint_dir):
                print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
                return
                
        elif args.output:
            checkpoint_dir = args.output
        else:
            # Create default name based on date and time
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"grufinity_{timestamp}"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    else:
        checkpoint_dir = ""
    
    # Synchronize checkpoint directory across processes if distributed
    if torch.distributed.is_initialized():
        if args.local_rank == 0:
            dir_tensor = torch.tensor([ord(c) for c in checkpoint_dir], dtype=torch.long).cuda()
            # Pad to fixed length
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            padded_dir[:len(dir_tensor)] = dir_tensor
        else:
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            
        # Broadcast from rank 0 to all processes
        torch.distributed.broadcast(padded_dir, 0)
        
        # Convert back to string on other ranks
        if args.local_rank != 0:
            nonzero_indices = padded_dir.nonzero().squeeze(-1)
            if len(nonzero_indices) > 0:
                str_len = nonzero_indices[-1].item() + 1
                checkpoint_dir = ''.join([chr(i) for i in padded_dir[:str_len].tolist()])
            else:
                checkpoint_dir = ""
    
    # Load model configuration from checkpoint or determine from arguments
    if resuming:
        # Load the checkpoint
        checkpoint_path = args.resume
        if os.path.isdir(checkpoint_path):
            # If directory, use the latest.pt file
            checkpoint_path = os.path.join(checkpoint_path, "latest.pt")
            
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            if args.local_rank == 0:
                print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
            return
            
        # Load checkpoint
        if args.local_rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model config from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            if args.local_rank == 0:
                print("Using model configuration from checkpoint")
        else:
            # Try to load from model_config.json in the same directory
            config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                if args.local_rank == 0:
                    print(f"Using model configuration from {config_path}")
            else:
                if args.local_rank == 0:
                    print("No model configuration found in checkpoint, using command line arguments")
                model_config = None
        
        # Get resume step
        if 'step' in checkpoint:
            resume_step = checkpoint['step']
            if args.local_rank == 0:
                print(f"Resuming from step {resume_step}")
        
        # Store loaded config for reference
        loaded_config = model_config
    
    # Determine model dimensions if not resuming or if no config found
    if not resuming or loaded_config is None:
        params_value = parse_size_with_suffix(args.params) if args.params is not None else None
        dim_value = int(parse_size_with_suffix(args.dim)) if args.dim is not None else None
        
        # Configure model based on parameters or explicit dimensions
        if params_value is not None:
            target_params = params_value
            
            if dim_value is not None and args.depth is None:
                # Dimension specified but not depth, solve for depth
                dim = round_to_multiple(dim_value)
                depth = solve_for_depth(
                    target_params, 
                    dim, 
                    256,  # vocab size
                    args.ff_mult,
                    args.expansion
                )
                if args.local_rank == 0:
                    print(f"Target params: {target_params/1e6:.1f}M, Dimension: {dim}, Calculated depth: {depth}")
            elif dim_value is None and args.depth is not None:
                # Depth specified but not dimension, solve for dimension
                depth = args.depth
                dim = solve_for_dimension(
                    target_params, 
                    depth, 
                    256,  # vocab size
                    args.ff_mult,
                    args.expansion
                )
                if args.local_rank == 0:
                    print(f"Target params: {target_params/1e6:.1f}M, Calculated dimension: {dim}, Depth: {depth}")
            else:
                # Scale both if neither or both are specified
                if dim_value is not None and args.depth is not None:
                    dim = round_to_multiple(dim_value)
                    depth = args.depth
                    if args.local_rank == 0:
                        print(f"Warning: Both dimension and depth specified with target params. Ignoring target params.")
                else:
                    # Calculate balanced depth/dimension based on parameter count
                    base_params = 15 * 1024 * 1024  # 15M reference
                    base_depth = 6
                    
                    if target_params >= base_params:
                        scaling_factor = (target_params / base_params) ** (1/3)
                        depth = max(base_depth, round(base_depth * scaling_factor))
                    else:
                        scaling_factor = (target_params / base_params) ** (1/4)
                        depth = max(2, round(base_depth * scaling_factor))
                    
                    # Solve for dimension with calculated depth
                    dim = solve_for_dimension(
                        target_params, 
                        depth, 
                        256,
                        args.ff_mult,
                        args.expansion
                    )
                    if args.local_rank == 0:
                        print(f"Target params: {target_params/1e6:.1f}M, Balanced - Dim: {dim}, Depth: {depth}")
        else:
            # Use explicit values from command line
            dim = round_to_multiple(dim_value) if dim_value is not None else 512
            depth = args.depth
        
        # Model configuration
        model_config = {
            "num_tokens": 256,  # byte-level tokenization
            "dim": dim,
            "depth": depth,
            "ff_mult": args.ff_mult,
            "expansion": args.expansion,
            "conv_kernel_size": 3,
            "use_lstm": False,
            "enable_conv": False,
            "dropout": 0.0
        }
    else:
        # Use the loaded model config
        dim = model_config["dim"]
        depth = model_config["depth"]
    
    # Calculate and display model size
    if args.local_rank == 0:
        param_count = calculate_model_size(model_config)
        print(f"Model size: {get_parameter_count_str(model_config)} parameters")
        print(f"Model configuration: {model_config}")
        
        # Save model configuration
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(model_config, f, indent=2)
    
    # Instantiate model with the determined configuration
    model = get_model(model_config)
    
    # 1) Setup optimizer - either AdamW or ScheduleFree
    if args.schedulefree:
        # Instantiate Schedule‑Free AdamW (no external scheduler needed)
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            betas=(args.sf_beta, 0.999),
            weight_decay=args.weight_decay
        )
        if args.local_rank == 0:
            print(f"Using ScheduleFree optimizer with lr={args.lr}, beta={args.sf_beta}")
    else:
        # Use standard AdamW (only when explicitly disabled)
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        if args.local_rank == 0:
            print(f"Using standard AdamW optimizer with lr={args.lr} (ScheduleFree disabled)")
    
    # 2) DeepSpeed config for tensor parallelism
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "tensor_parallel": {
            "tp": {
                "tp_size": args.tp_size,
                "tp_grain_size": 64
            }
        }
    }
    
    # 3) Initialize DeepSpeed engine with explicit config (no CLI args)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        model_parameters=model.parameters(),
        dist_init_required=True
    )
    
    # 3.1) Load optimizer state if resuming
    if resuming and 'optimizer_state_dict' in checkpoint:
        if args.local_rank == 0:
            print("Loading optimizer state from checkpoint")
        
        # Force learning rate if requested
        if args.force_lr:
            old_lr = optimizer.param_groups[0]['lr']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            if args.local_rank == 0:
                print(f"Forced learning rate from {old_lr} to {args.lr}")
        else:
            # Just load the state as-is
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.local_rank == 0:
                print(f"Resumed with learning rate: {optimizer.param_groups[0]['lr']}")
    
    # 3.2) Load model weights if resuming
    if resuming and 'model_state_dict' in checkpoint:
        if args.local_rank == 0:
            print("Loading model weights from checkpoint")
        model_engine.module.load_state_dict(checkpoint['model_state_dict'])
    
    # 4) Put ScheduleFree optimizer into train mode if used
    if args.schedulefree:
        model_engine.optimizer.train()
    
    # 5) Prepare Datasets & Samplers
    # Use 10K samples per epoch as a reasonable default to limit memory usage
    samples_per_epoch = min(10000, train_steps * batch_size)
    
    # Create the base dataset
    base_dataset = MemoryMappedDataset(
        filepath=args.data,
        seq_len=seq_len,
        samples_per_epoch=samples_per_epoch
    )
    
    # Create train and validation datasets using modulo-based splitting
    # Validation set: indices where idx % val_mod == 0
    # Training set: all other indices
    train_dataset = ModuloSplitDataset(base_dataset, args.val_mod, remainder=1)
    val_dataset = ModuloSplitDataset(base_dataset, args.val_mod, remainder=0)
    
    if args.local_rank == 0:
        print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create samplers for both datasets
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=model_engine.world_size,
        rank=model_engine.global_rank,
        shuffle=True,
        seed=SEED
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=model_engine.world_size,
        rank=model_engine.global_rank,
        shuffle=False,
        seed=SEED
    )
    
    # 6) DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=1,  # Reduced to prevent multiple copies of memory map
        pin_memory=True,
        worker_init_fn=lambda wid: torch.manual_seed(SEED + wid),
        persistent_workers=False  # Avoid persistent workers to prevent hanging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=lambda wid: torch.manual_seed(SEED + 100 + wid),
        persistent_workers=False
    )
    
    # Create metrics log file
    if model_engine.global_rank == 0 and checkpoint_dir:
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        with open(metrics_log_path, 'w') as f:
            header = [
                "step", "time", "train_loss", "val_loss", 
                "tokens_processed", "tokens_per_sec", "learning_rate", "batch_size"
            ]
            f.write('\t'.join(header) + '\n')
    
    # Helper function to save checkpoint
    def save_checkpoint(step, train_loss, val_loss=None):
        if model_engine.global_rank != 0 or not checkpoint_dir:
            return
            
        checkpoint = {
            'step': step,
            'model_state_dict': model_engine.module.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_config': model_config,
            'args': vars(args)
        }
        
        # Save optimizer state
        if args.schedulefree:
            checkpoint['optimizer_state_dict'] = model_engine.optimizer.state_dict()
        else:
            # Standard optimizer
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Create filename with step and loss info
        val_info = f"-val_{val_loss:.4f}" if val_loss is not None else ""
        filename = f"minlm-step-{step:05d}-loss-{train_loss:.4f}{val_info}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save model config for easy access
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        return checkpoint_path
    
    # Helper function for validation
    def validate():
        model_engine.eval()
        total_loss = 0.0
        batch_count = 0
        val_token_count = 0
        validation_start = time.time()
        
        # Switch ScheduleFree to eval mode
        if args.schedulefree:
            model_engine.optimizer.eval()
        
        if model_engine.global_rank == 0:
            print("\nRunning validation...")
            val_pbar = tqdm(total=5, desc="Validation", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        with torch.no_grad():
            # Set fixed epoch for validation to ensure deterministic behavior
            val_sampler.set_epoch(0)
            
            for x in val_loader:
                inputs, targets = x[:, :-1], x[:, 1:]
                inputs, targets = inputs.to(model_engine.device), targets.to(model_engine.device)
                
                # Forward pass
                loss = model_engine(inputs, return_loss=True)
                
                # Accumulate loss
                total_loss += loss.item()
                batch_count += 1
                val_token_count += inputs.numel()
                
                # Update progress bar
                if model_engine.global_rank == 0:
                    val_pbar.update(1)
                
                # Limit validation to a few batches for speed
                if batch_count >= 5:
                    break
        
        if model_engine.global_rank == 0:
            val_pbar.close()
            validation_time = time.time() - validation_start
            val_tokens_per_sec = val_token_count / validation_time if validation_time > 0 else 0
            print(f"Validation complete: {val_tokens_per_sec:.2f} tokens/sec")
        
        # Switch back to train mode
        model_engine.train()
        if args.schedulefree:
            model_engine.optimizer.train()
        
        # Calculate average
        avg_loss = total_loss / max(1, batch_count)
        return avg_loss
    
    # Helper function to log metrics
    def log_metrics(step, train_loss, val_loss=None):
        if model_engine.global_rank != 0 or not checkpoint_dir:
            return
            
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        elapsed = time.time() - start_time
        
        # Calculate tokens processed and tokens per second
        tokens_processed = step * batch_size * seq_len
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        
        current_lr = model_engine.optimizer.param_groups[0]['lr']
        
        values = [
            str(step),
            f"{elapsed:.2f}",
            f"{train_loss:.6f}",
            str(val_loss if val_loss is not None else "NA"),
            str(tokens_processed),
            f"{tokens_per_sec:.2f}",
            f"{current_lr:.8f}",
            str(batch_size)
        ]
        
        with open(metrics_log_path, 'a') as f:
            f.write('\t'.join(values) + '\n')
    
    # 7) Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    total_tokens_processed = 0
    
    # Adjust starting step if resuming
    start_step = resume_step if resuming else 0
    
    # Create progress bar if primary process
    if model_engine.global_rank == 0:
        pbar = tqdm(total=train_steps, desc="Training", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        pbar.update(start_step)  # Update for resumed progress
        
    for step in range(start_step, start_step + train_steps):
        # Set epoch for deterministic shuffling
        epoch = step // len(train_loader)
        train_sampler.set_epoch(epoch)
        
        # Iterate through data loader for this step
        for batch_idx, x in enumerate(train_loader):
            # Split into input and target
            inputs, targets = x[:, :-1], x[:, 1:]
            
            # Move to device
            inputs, targets = inputs.to(model_engine.device), targets.to(model_engine.device)
            
            # Forward pass (returns loss directly)
            loss = model_engine(inputs, return_loss=True)
            
            # Backward pass
            model_engine.backward(loss)
            
            # Optimizer step
            model_engine.step()
            
            # Log progress
            if model_engine.global_rank == 0 and batch_idx == 0:
                # Update tokens processed
                total_tokens_processed += batch_size * seq_len
                
                # Calculate tokens per second
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
                
                # Update progress bar with stats
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{model_engine.optimizer.param_groups[0]['lr']:.6f}",
                    'tok/s': f"{tokens_per_sec:.2f}"
                })
                pbar.update(1)
                
                # Log metrics for training
                log_metrics(step, loss.item())
            
            # Break after one batch per step
            break
        
        # Validate periodically
        if args.validate_every > 0 and step > 0 and step % args.validate_every == 0:
            val_loss = validate()
            
            if model_engine.global_rank == 0:
                print(f"[Step {step:03d}] Validation Loss: {val_loss:.4f}")
                
                # Log metrics with validation
                log_metrics(step, loss.item(), val_loss)
                
                # Check if this is best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(step, loss.item(), val_loss)
                    print(f"New best model saved at step {step} with validation loss {val_loss:.4f}")
        
        # Save checkpoint periodically
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_path = save_checkpoint(step, loss.item())
            if model_engine.global_rank == 0:
                print(f"Checkpoint saved to {save_path}")
    
    # Final validation
    final_val_loss = validate()
    
    # Save final checkpoint
    if model_engine.global_rank == 0:
        final_path = save_checkpoint(train_steps, loss.item(), final_val_loss)
        
        # After training, switch to eval mode if using ScheduleFree
        if args.schedulefree:
            model_engine.optimizer.eval()
        
        # Close progress bar
        pbar.close()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        tokens_per_sec = total_tokens_processed / total_time if total_time > 0 else 0
        
        # Print training summary
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Processed {total_tokens_processed:,} tokens at {tokens_per_sec:.2f} tokens/sec")
        print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    main()

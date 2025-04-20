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
from datetime import timedelta
import sys
import asyncio
import shutil
from tqdm import tqdm
from schedulefree import AdamWScheduleFree

# Set NCCL environment variables to help with distributed training issues
os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # Avoid certain interfaces
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand (use TCP instead)
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable GPU Direct P2P if causing issues

# Make sure the environment variables are actually applied
for var in ["NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE"]:
    if var in os.environ:
        print(f"{var}={os.environ[var]}")

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

class InfiniteRandomSamplingDataset(Dataset):
    """Dataset that infinitely samples random chunks from a file without loading it entirely"""
    def __init__(self, filepath, seq_len, seed=42, virtual_size=1000000):
        super().__init__()
        self.filepath = filepath
        self.seq_len = seq_len
        
        # Get file size
        self.file_size = os.path.getsize(filepath)
        
        # Maximum valid starting position
        self.max_start = max(0, self.file_size - seq_len - 1)
        
        # Virtual size - doesn't affect actual sampling, just affects how DataLoader
        # perceives the dataset size for batching purposes
        self.virtual_size = virtual_size
        
        # Global RNG for consistent behavior
        self.global_rng = random.Random(seed)
        
        # Thread-local storage for file handles
        import threading
        self.local = threading.local()
        
        print(f"InfiniteRandomSamplingDataset: Using file {filepath} ({self.file_size:,} bytes)")
        print(f"Virtual size: {self.virtual_size:,} (purely for iteration control)")
        print(f"Data sampled randomly on demand - no epoch boundaries - no memory loading")
    
    def __len__(self):
        # Return virtual size - this doesn't affect the actual sampling
        # It just controls how DataLoader perceives the dataset size
        return self.virtual_size
    
    def _get_file_handle(self):
        """Get file handle, creating it if needed (thread-local)"""
        if not hasattr(self.local, 'file') or self.local.file is None or self.local.file.closed:
            self.local.file = open(self.filepath, 'rb')
        return self.local.file
    
    def __getitem__(self, idx):
        # Completely ignore idx - just sample randomly
        # Create thread-local RNG for better parallelism
        if not hasattr(self.local, 'rng'):
            # Each thread gets its own RNG, seeded from the global one
            thread_seed = self.global_rng.randint(0, 2**32-1)
            self.local.rng = random.Random(thread_seed)
        
        # Sample a completely random position
        start_pos = self.local.rng.randint(0, self.max_start)
        
        try:
            # Get file handle and read data
            f = self._get_file_handle()
            f.seek(start_pos)
            data = f.read(self.seq_len + 1)  # +1 for target token
            
            # Convert to tensor - use bytearray for efficiency
            tensor = torch.tensor(bytearray(data), dtype=torch.long)
            
            # Handle edge case
            if tensor.size(0) < self.seq_len + 1:
                padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
                tensor = torch.cat([tensor, padding])
                
            return tensor
        except Exception as e:
            print(f"Error reading from file at position {start_pos}: {e}")
            # Try to recover file handle
            try:
                if hasattr(self.local, 'file') and self.local.file:
                    self.local.file.close()
                    self.local.file = None
            except:
                pass
            return torch.zeros(self.seq_len + 1, dtype=torch.long)
    
    def __del__(self):
        """Clean up resources - need a different approach for thread-local storage"""
        try:
            # Check if local attribute exists and has a file
            if hasattr(self, 'local') and hasattr(self.local, 'file') and self.local.file:
                self.local.file.close()
        except:
            pass

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
        description='DeepSpeed Tensor Parallel Training for minLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )
    
    # Distributed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--exclude_gpus', type=str, default="",
                        help='comma-separated list of GPU indices to exclude (e.g., "0,3")')
    
    # Training arguments
    parser.add_argument('--train_steps', type=str, default="100",
                        help='number of training steps')
    parser.add_argument('--port', type=int, default=29500,
                        help='port for distributed communication')
    parser.add_argument('--data', type=str, required=True,
                        help='path to training data file')
    parser.add_argument('--val_mod', type=int, default=10,
                        help='modulo for validation set (1/10th for validation)')
    parser.add_argument('--output', type=str, default=None,
                        help='directory to save checkpoints (auto-generated if not specified)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save checkpoint every N steps')
    parser.add_argument('--validate_every', type=int, default=50,
                        help='validate every N steps')
    parser.add_argument('--force_lr', action='store_true',
                        help='force use command line learning rate when resuming')
    
    # Model configuration - Create a group to track explicitly provided arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--dim', type=str, default="512", 
                        help='model hidden dimension (auto-calculated if --params is used)')
    model_group.add_argument('--depth', type=int, default=6,
                        help='number of transformer layers (auto-calculated if --params is used)')
    model_group.add_argument('--seq_len', type=str, default="128",
                        help='sequence length for training')
    model_group.add_argument('--params', type=str, default=None,
                        help='target parameter count (e.g., 15m, 100m, 1g) - conflicts with setting both --dim and --depth')
    model_group.add_argument('--ff_mult', type=float, default=4.0,
                        help='feedforward multiplier')
    model_group.add_argument('--expansion', type=float, default=1.5,
                        help='expansion factor for minGRU')
    
    # Optimizer configuration
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--batch_size', type=str, default="4",
                        help='batch size per GPU')
    parser.add_argument('--no-schedulefree', dest='schedulefree', action='store_false', default=True,
                        help='disable ScheduleFree optimizer (default: enabled)')
    parser.add_argument('--sf_beta', type=float, default=0.9,
                        help='ScheduleFree beta parameter')
    
    # Add checkpoint keeping parameter
    parser.add_argument('--keep_checkpoints', type=int, default=3,
                        help='number of recent checkpoints to keep (default: 3)')
    
    # Parse args first to get all defaults filled in
    args = parser.parse_args()
    
    # Add a special attribute to track which arguments were explicitly provided
    # This will help us distinguish between default values and user-provided values
    args._explicitly_set = []
    
    # Get the command-line arguments actually passed
    import sys
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            # Extract the arg name without the -- prefix
            arg_name = arg[2:].split('=')[0]
            args._explicitly_set.append(arg_name)
    
    return args

def synchronize_processes():
    """Synchronize all distributed processes with better error handling and staggered approach"""
    # First check if distributed is initialized to avoid errors
    if not torch.distributed.is_initialized():
        print("Warning: Skipping synchronization as distributed is not yet initialized")
        return
        
    try:
        # Get current device and rank info
        current_device = torch.cuda.current_device()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        if rank == 0:
            print(f"Waiting for all {world_size} processes to synchronize... (using device {current_device})")
        
        # Print device health status
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > current_device
        if not cuda_ok:
            print(f"WARNING: Rank {rank} has CUDA device issues. Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")
        
        # Staggered approach: small delay based on rank to avoid thundering herd
        time.sleep(0.01 * rank)
        
        # Simple barrier with device_ids to prevent hanging
        torch.distributed.barrier(device_ids=[current_device])
        
        # Sleep a bit after barrier to let things settle
        time.sleep(0.01)
        
        if rank == 0:
            print("All processes synchronized successfully")
            
    except Exception as e:
        print(f"ERROR in synchronize_processes: {e}")
        # Try to proceed anyway

def verify_gpu_health():
    """Verify that all GPUs are working properly"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA devices")
        
        # Check each device
        for i in range(device_count):
            try:
                # Try to allocate and free a small tensor on each device
                device = torch.device(f'cuda:{i}')
                test_tensor = torch.zeros((10, 10), device=device)
                del test_tensor
                torch.cuda.empty_cache()
                print(f"GPU {i} check: OK - {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"GPU {i} check: FAILED - {e}")
                
        # GPU 0 is particularly important for distributed training
        if device_count > 0:
            try:
                # More intensive test for GPU 0
                device = torch.device('cuda:0')
                a = torch.rand((1000, 1000), device=device)
                b = torch.rand((1000, 1000), device=device)
                c = torch.matmul(a, b)
                del a, b, c
                torch.cuda.empty_cache()
                print("GPU 0 matrix multiplication test: OK")
            except Exception as e:
                print(f"GPU 0 matrix multiplication test: FAILED - {e}")

def main():
    args = get_args()
    
    # Set custom port for distributed communication
    if args.port != 29500:
        os.environ['MASTER_PORT'] = str(args.port)
        
    # Handle GPU exclusion if specified
    if args.exclude_gpus:
        exclude_list = [int(x.strip()) for x in args.exclude_gpus.split(',')]
        if args.local_rank == 0:
            print(f"Excluding GPUs: {exclude_list}")
        
        # If current GPU is in exclude list, disable CUDA for this process
        if args.local_rank in exclude_list:
            print(f"Rank {args.local_rank} was specified to be excluded, forcing CPU execution")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Verify GPU health before proceeding
    if args.local_rank == 0:
        verify_gpu_health()
    
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
            print(f"Broadcasting checkpoint directory: {checkpoint_dir}")
            dir_tensor = torch.tensor([ord(c) for c in checkpoint_dir], dtype=torch.long).cuda()
            # Pad to fixed length
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            padded_dir[:len(dir_tensor)] = dir_tensor
        else:
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            
        # Broadcast from rank 0 to all processes with explicit device
        current_device = torch.cuda.current_device()
        torch.distributed.broadcast(padded_dir, 0, device_ids=[current_device])
        
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
            
            # Format the parameter count for display
            if target_params >= 1e9:
                param_display = f"{target_params/1e9:.1f}B"
            else:
                param_display = f"{target_params/1e6:.1f}M"
            
            # Check which args were explicitly set by the user (not just defaults)
            dim_explicitly_set = 'dim' in getattr(args, '_explicitly_set', [])
            depth_explicitly_set = 'depth' in getattr(args, '_explicitly_set', [])
            
            if args.local_rank == 0:
                print(f"Explicit dimension set: {dim_explicitly_set}, Explicit depth set: {depth_explicitly_set}")
            
            if dim_explicitly_set and not depth_explicitly_set:
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
                    print(f"Target params: {param_display}, Fixed dimension: {dim}, Calculated depth: {depth}")
                    
            elif not dim_explicitly_set and depth_explicitly_set:
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
                    print(f"Target params: {param_display}, Fixed depth: {depth}, Calculated dimension: {dim}")
                    
            elif dim_explicitly_set and depth_explicitly_set:
                # Error if both dimension and depth are specified along with params
                dim = round_to_multiple(dim_value)
                depth = args.depth
                if args.local_rank == 0:
                    print(f"ERROR: Cannot specify dimension ({dim}), depth ({depth}), AND target parameters ({param_display}).")
                    print(f"Please specify either dimension, depth, or neither - but not both when using --params.")
                    sys.exit(1)
            else:
                # Neither dimension nor depth specified, calculate both based on target params
                # Use a balanced scaling approach similar to ../gruf/learn.py
                
                # Different scaling approaches based on model size for better scaling
                if target_params < 15 * 1024 * 1024:  # < 15M params (small models)
                    # Small models should be shallower with reasonable width
                    base_params = 15 * 1024 * 1024
                    base_depth = 6
                    base_dim = 512
                    
                    # Calculate balanced depth & dim
                    scaling = (target_params / base_params) ** 0.25  # Less aggressive scaling for small models
                    depth = max(2, round(base_depth * scaling))  # Min depth of 2
                    dim = solve_for_dimension(target_params, depth, 256, args.ff_mult, args.expansion)
                    
                elif target_params < 100 * 1024 * 1024:  # < 100M params (medium models)
                    # Medium models get balanced scaling
                    base_params = 15 * 1024 * 1024
                    base_depth = 6
                    base_dim = 512
                    
                    # Calculate balanced depth & dim with priority on depth
                    scaling = (target_params / base_params) ** 0.33  # Cube root scaling
                    depth = max(6, round(base_depth * scaling * 1.2))  # Favor depth (20% more)
                    dim = solve_for_dimension(target_params, depth, 256, args.ff_mult, args.expansion)
                    
                else:  # Large models (≥ 100M params)
                    # For large models, scale depth more aggressively
                    base_params = 100 * 1024 * 1024
                    base_depth = 12  # Start from a deeper base
                    base_dim = 768
                    
                    # Calculate balanced depth & dim with higher priority on depth
                    scaling = (target_params / base_params) ** 0.4  # Emphasis on depth for large models
                    depth = max(12, round(base_depth * scaling * 1.3))  # Favor depth strongly (30% more)
                    dim = solve_for_dimension(target_params, depth, 256, args.ff_mult, args.expansion)
                
                # Ensure dimension is a multiple of 64
                dim = round_to_multiple(dim, 64)
                
                if args.local_rank == 0:
                    print(f"Target params: {param_display}, Balanced scaling - Depth: {depth}, Dimension: {dim}")
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
        
        # Additional verification to ensure the calculated parameters match target
        if params_value is not None and not resuming:
            # Check if there's a significant discrepancy
            if abs(param_count - params_value) / params_value > 0.05:  # More than 5% difference
                print(f"WARNING: Calculated parameter count ({param_count:,}) differs from target ({params_value:,})")
                print(f"Difference: {abs(param_count - params_value) / params_value * 100:.1f}%")
                print("Adjusting model dimensions to better match target parameter count...")
                
                # Adjust the dimension (keeping depth fixed) to match parameter count more precisely
                orig_dim = model_config["dim"]
                orig_depth = model_config["depth"]
                
                # Recalculate dimension with fixed depth to match target params precisely
                dim = solve_for_dimension(params_value, orig_depth, 256, args.ff_mult, args.expansion)
                
                # Update model config
                model_config["dim"] = dim
                
                # Recalculate and display
                param_count = calculate_model_size(model_config)
                print(f"Adjusted dimension from {orig_dim} to {dim} (depth stays at {orig_depth})")
        
        print(f"Model size: {get_parameter_count_str(model_config)} parameters ({param_count:,})")
        print(f"Model configuration: {model_config}")
        
        # Save model configuration to both config.json files
        if checkpoint_dir:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"Saved model configuration to {config_path}")
    
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
    
    # 3) Initialize DeepSpeed engine with explicit config and better error handling
    try:
        # No barrier before DeepSpeed init - DeepSpeed handles this internally
        print(f"Rank {args.local_rank}: Initializing DeepSpeed (device: {torch.cuda.current_device()})")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            model_parameters=model.parameters(),
            dist_init_required=True
        )
        print(f"Rank {args.local_rank}: DeepSpeed initialization successful")
    except Exception as e:
        print(f"ERROR in DeepSpeed initialization (rank {args.local_rank}): {e}")
        # Try to recover
        if torch.distributed.is_initialized():
            print(f"Rank {args.local_rank}: Trying to synchronize after error...")
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            print(f"Rank {args.local_rank}: Synchronization after error complete")
        raise
    
    # Synchronize after DeepSpeed initialization to ensure all processes are ready
    synchronize_processes()
    
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
        
        # Synchronize after loading checkpoint to ensure all processes have updated weights
        synchronize_processes()
    
    # 4) Put ScheduleFree optimizer into train mode if used
    if args.schedulefree:
        model_engine.optimizer.train()
    
    # Create the datasets using our infinite random sampler
    # The same file is used for both training and validation
    train_dataset = InfiniteRandomSamplingDataset(
        filepath=args.data,
        seq_len=seq_len,
        seed=SEED,
        # Virtual size determines when DataLoader thinks it's done with an "epoch"
        # This is just for iteration control, not actual sampling boundaries
        virtual_size=10000  # ~10k batches per "epoch" for DataLoader
    )
    
    # Create a separate instance for validation with a different seed
    val_dataset = InfiniteRandomSamplingDataset(
        filepath=args.data,
        seq_len=seq_len,
        seed=SEED + 100,  # Different seed for validation
        virtual_size=500  # Smaller for validation
    )
    
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
    
    # Keep track of recent checkpoint files and what best points to
    recent_checkpoints = []
    best_checkpoint_filename = None

    # Helper function to save checkpoint - now with symlinks for both best and latest
    def save_checkpoint(step, train_loss, val_loss=None, is_best=False):
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
        
        # Create a standardized filename with step and loss info
        # Always include validation loss in filename if available
        if val_loss is not None:
            filename = f"minlm-step-{step:05d}-loss-{train_loss:.4f}-val_{val_loss:.4f}.pt"
        else:
            filename = f"minlm-step-{step:05d}-loss-{train_loss:.4f}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Save checkpoint atomically using temporary file (this is the only full save)
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Create symlink for latest.pt (remove existing one if it exists)
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path) or os.path.islink(latest_path):
            try:
                os.remove(latest_path)
            except Exception as e:
                print(f"Warning: Failed to remove existing latest.pt: {e}")
        
        # Create symlink from latest.pt to the current checkpoint
        try:
            os.symlink(os.path.basename(checkpoint_path), latest_path)
            print(f"Created symlink: latest.pt -> {os.path.basename(checkpoint_path)}")
        except Exception as e:
            # Fall back to copy if symlink fails (e.g., on platforms without symlinks)
            print(f"Warning: Failed to create symlink, falling back to copy: {e}")
            shutil.copy2(checkpoint_path, latest_path)
        
        # Save model config for easy access
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Handle checkpoint rotation
        nonlocal recent_checkpoints, best_checkpoint_filename
        
        # Add current checkpoint to tracking list
        recent_checkpoints.append(checkpoint_path)
        
        if is_best:
            # Create symlink for best checkpoint
            best_path = os.path.join(checkpoint_dir, f"best.pt")
            
            # Remove existing best symlink if it exists
            if os.path.exists(best_path) or os.path.islink(best_path):
                try:
                    os.remove(best_path)
                    print(f"Removed previous best symlink")
                except Exception as e:
                    print(f"Warning: Failed to remove previous best symlink: {e}")
            
            # Create symlink from best.pt to the current checkpoint
            try:
                os.symlink(os.path.basename(checkpoint_path), best_path)
                print(f"Created best symlink: best.pt -> {os.path.basename(checkpoint_path)}")
                
                # Double-check that best.pt exists
                if os.path.exists(best_path) or os.path.islink(best_path):
                    print(f"Confirmed best.pt exists and points to {os.path.basename(checkpoint_path)}")
                else:
                    print(f"WARNING: best.pt could not be verified after creation!")
                    # Fallback to direct copy if symlink doesn't seem to exist
                    shutil.copy2(checkpoint_path, best_path)
                    print(f"Fallback: Copied checkpoint directly to best.pt")
            except Exception as e:
                print(f"Warning: Failed to create best symlink, falling back to copy: {e}")
                shutil.copy2(checkpoint_path, best_path)
                print(f"Directly copied checkpoint to best.pt instead of symlinking")
            
            # Track which checkpoint is currently the best
            best_checkpoint_filename = os.path.basename(checkpoint_path)
        
        # Clean up old checkpoints, making sure not to delete the best one
        # Get the number of checkpoints to keep from args
        num_to_keep = args.keep_checkpoints
        
        # Keep only the N most recent checkpoints (plus don't delete best)
        while len(recent_checkpoints) > num_to_keep:
            # Find the oldest checkpoint that's not the best
            delete_candidate = None
            for i, old_checkpoint in enumerate(recent_checkpoints):
                old_checkpoint_filename = os.path.basename(old_checkpoint)
                if old_checkpoint_filename != best_checkpoint_filename:
                    delete_candidate = old_checkpoint
                    delete_index = i
                    break
            
            # If we couldn't find a non-best checkpoint to delete, break out of the loop
            if delete_candidate is None:
                print("All remaining checkpoints are marked as best, keeping them all")
                break
                
            # Remove the checkpoint from our tracking list
            recent_checkpoints.pop(delete_index)
            
            # Delete the file if it exists
            if os.path.exists(delete_candidate):
                try:
                    os.remove(delete_candidate)
                    print(f"Removed old checkpoint: {os.path.basename(delete_candidate)}")
                except Exception as e:
                    print(f"Warning: Failed to remove old checkpoint: {e}")
        
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
            # No need to set epoch - validation dataset has its own fixed seed
            
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
        
        # Calculate average
        avg_loss = total_loss / max(1, batch_count)
        
        # Synchronize all processes after validation to prevent some processes 
        # from continuing while others are still validating
        synchronize_processes()
        
        # Switch back to train mode
        model_engine.train()
        if args.schedulefree:
            model_engine.optimizer.train()
        
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
    
    # Verify model configuration before training
    if model_engine.global_rank == 0:
        actual_params = sum(p.numel() for p in model_engine.module.parameters())
        print(f"\nModel verification:")
        print(f"- Configured parameters: {calculate_model_size(model_config):,}")
        print(f"- Actual parameters: {actual_params:,}")
        print(f"- Model dimension: {model_config['dim']}")
        print(f"- Model depth: {model_config['depth']}")
        print(f"- Tensor parallelism: {args.tp_size} GPUs")
        print(f"- Keeping {args.keep_checkpoints} most recent checkpoints")
        
        # Log the configuration summary
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "model_summary.txt"), "w") as f:
                f.write(f"Training command: {' '.join(sys.argv)}\n")
                f.write(f"Model parameters: {actual_params:,}\n")
                f.write(f"Configuration: {json.dumps(model_config, indent=2)}\n")
                f.write(f"Distributed setup: {args.tp_size} GPUs with tensor parallelism\n")
                f.write(f"Sequence length: {seq_len}\n")
                f.write(f"Batch size: {batch_size} per GPU\n")
    
    # Synchronize all processes before starting training loop
    synchronize_processes()
    
    # Create progress bar if primary process
    if model_engine.global_rank == 0:
        pbar = tqdm(total=train_steps, desc="Training", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        pbar.update(start_step)  # Update for resumed progress
        
    for step in range(start_step, start_step + train_steps):
        # We don't need to set epochs anymore - the dataset is truly random
        # Just let it run indefinitely
        
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
        val_loss_for_checkpoint = None
        if args.validate_every > 0 and step > 0 and step % args.validate_every == 0:
            val_loss = validate()
            val_loss_for_checkpoint = val_loss
            
            if model_engine.global_rank == 0:
                print(f"[Step {step:03d}] Validation Loss: {val_loss:.4f}")
                
                # Log metrics with validation
                log_metrics(step, loss.item(), val_loss)
                
                # Check if this is best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Mark as best but don't save yet - we'll save in the checkpoint step below
                    print(f"New best model found at step {step} with validation loss {val_loss:.4f}")
        
        # Save checkpoint periodically
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            # Always save on checkpoint steps, with validation loss if available
            is_best = False
            
            # Only consider this checkpoint as best if we just did validation
            if val_loss_for_checkpoint is not None:
                is_best = val_loss_for_checkpoint < best_val_loss
                if is_best:
                    # Update the best val loss for tracking
                    best_val_loss = val_loss_for_checkpoint
                    if model_engine.global_rank == 0:
                        print(f"New best model found with validation loss {val_loss_for_checkpoint:.4f}")
            
            save_path = save_checkpoint(step, loss.item(), val_loss_for_checkpoint, is_best=is_best)
            
            if model_engine.global_rank == 0:
                if is_best:
                    print(f"Best model checkpoint saved to {save_path} (val loss: {val_loss_for_checkpoint:.4f})")
                else:
                    print(f"Checkpoint saved to {save_path}")
    
    # Final validation
    final_val_loss = validate()
    
    # Save final checkpoint and make it the best if it's better than previous best
    if model_engine.global_rank == 0:
        # Save the final checkpoint with validation loss and check if it's the best one
        is_final_best = final_val_loss < best_val_loss
        
        # Update best val loss if this is the best
        if is_final_best:
            best_val_loss = final_val_loss
            print(f"Final model is best model with validation loss {final_val_loss:.4f}")
            
        final_path = save_checkpoint(train_steps + start_step, loss.item(), final_val_loss, is_best=is_final_best)
        
        if is_final_best:
            print(f"Best model saved to best.pt -> {os.path.basename(final_path)}")
        
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

import os, random, numpy as np, hashlib
# Backend will be selected via command-line arguments (--cuda or --rocm)
import torch, torch.nn as nn
import torch.distributed as dist
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
import shutil
from tqdm import tqdm
from schedulefree import AdamWScheduleFree

def configure_backend(args):
    """Configure environment for the selected backend (CUDA or ROCm)"""
    use_rocm = args.rocm
    
    print(f"Using {'ROCm' if use_rocm else 'CUDA'} backend")
    
    # Only handle MIOpen cache setup which is Python-specific (for ROCm)
    if use_rocm and "SLURM_NODEID" in os.environ:
        os.environ["MIOPEN_USER_DB_PATH"] = f"/tmp/{os.environ.get('USER', 'user')}-miopen-cache-{os.environ['SLURM_NODEID']}"
        os.environ["MIOPEN_SYSTEM_DB_PATH"] = os.environ["MIOPEN_USER_DB_PATH"]
        print(f"MIOpen cache path: {os.environ['MIOPEN_USER_DB_PATH']}")
    
    # Set higher precision for float32 matrix multiplication (CUDA only) 
    # This enables TensorFloat32 on supported NVIDIA GPUs
    if not use_rocm:
        torch.set_float32_matmul_precision('high')
        print("Enabled high precision matrix multiplication (TensorFloat32)")
    
    return use_rocm

def debug_distributed_info():
    """Print debug information about the distributed environment"""
    # This will be called after DeepSpeed initialization
    print(f"\n----- Distributed Environment Debug Info -----")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}")
    print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'Not set')}")
    print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU Device Count: {device_count}")
        current_device = torch.cuda.current_device()
        print(f"Current Device: {current_device} ({torch.cuda.get_device_name(current_device)})")
    
    if torch.distributed.is_initialized():
        print(f"Distributed is initialized:")
        print(f"  - World size: {torch.distributed.get_world_size()}")
        print(f"  - Rank: {torch.distributed.get_rank()}")
        print(f"  - Backend: {torch.distributed.get_backend()}")
    else:
        print(f"Distributed is NOT initialized")
        
    print(f"------------------------------------------\n")

# Explicitly set port for torch.distributed to avoid conflicts
# This ensures we're using the same port as passed in MASTER_PORT environment variable
if 'MASTER_PORT' in os.environ:
    os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
else:
    os.environ['MASTER_PORT'] = '3442'  # Fallback to our fixed port
    
print(f"Using MASTER_PORT={os.environ['MASTER_PORT']}")

# Disable torch.distributed direct initialization to allow DeepSpeed to handle it
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # Enable detailed distributed logging

# MIOpen cache setup is now handled in configure_backend()

# TensorFloat32 precision is now handled in configure_backend()

# Import the minLM model
from mingru.minLM import minLM

# Add imports for evolutionary gossip protocol
import logging
from gossip import EvolutionaryTrainingNode

# 1) Deterministic seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():  # Works for both CUDA and ROCm
    torch.cuda.manual_seed_all(SEED)

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

class ContinuousIIDDataset(Dataset):
    """
    Dataset that samples across the entire file using true IID (Independent and Identically Distributed) 
    sampling with replacement. Uses a single, continuous random number generator that never resets, 
    ensuring there are no artificial distribution shifts at arbitrary epoch boundaries.
    
    Memory-efficient: No pre-materialization of indices, works with datasets of any size.
    """
    def __init__(self, filepath, seq_len, seed=42, samples_per_epoch=None, batch_size=1, 
                 log_sample_hashes=False, checkpoint_dir=None):
        super().__init__()
        self.filepath = filepath
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size
        
        # Use memory mapping for faster file access
        self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        
        # Maximum valid starting position
        self.max_start = len(self.mmap) - seq_len - 1
        
        # Set samples per epoch directly (used only for calculating total length)
        if samples_per_epoch is None:
            # Default to 100 batches per epoch
            self.samples_per_epoch = 100 * self.batch_size
        else:
            self.samples_per_epoch = samples_per_epoch
            
        # Thread-local storage for file handles
        import threading
        self.local = threading.local()
        
        # Create a SINGLE continuous RNG that will never be reset
        # This ensures true IID sampling across all steps with no distribution shifts
        self.rng = random.Random(self.seed)
        
        # Calculate how many unique positions we can sample from
        self.unique_positions = self.max_start + 1
        
        # Print debug info about file size and sampling range
        # Only print from rank 0 if in distributed environment, otherwise always print
        should_print = True
        # Safely get model_engine from globals if it exists
        model_engine_obj = globals().get('model_engine') if 'model_engine' in globals() else None
        if model_engine_obj is not None and hasattr(model_engine_obj, 'global_rank'):
            should_print = (model_engine_obj.global_rank == 0)
            
        if should_print:
            print(f"DEBUG: File size: {len(self.mmap):,} bytes")
            print(f"DEBUG: Max start position: {self.max_start:,}")
            print(f"DEBUG: Random sampling range type: {type(self.max_start)}")
        
        # Calculate batches per epoch (used only for reporting)
        self.batches_per_epoch = self.samples_per_epoch // self.batch_size
        
        # Setup sample logging if requested
        self.log_sample_hashes = log_sample_hashes
        self.checkpoint_dir = checkpoint_dir
        self.sample_counter = 0
        
        if self.log_sample_hashes and self.checkpoint_dir:
            self.sample_hash_file = os.path.join(self.checkpoint_dir, "sample_distribution.tsv")
            # Create/clear the sample hash log file with a header
            with open(self.sample_hash_file, 'w') as f:
                f.write("step_idx\tfile_pos\thash\n")
        
        print(f"ContinuousIIDDataset: Using file {filepath} ({len(self.mmap):,} bytes)")
        print(f"File contains approximately {self.unique_positions:,} possible unique samples")
        print(f"Training with {self.batches_per_epoch} batches per epoch ({self.samples_per_epoch} samples)")
        print(f"Using true continuous IID sampling with NO resets at epoch boundaries")
        print(f"Memory efficient: Only loading individual samples on-demand, no pre-materialization")
    
    def __len__(self):
        return self.samples_per_epoch
    
    def set_epoch(self, epoch):
        """
        No-op method kept for API compatibility.
        With continuous IID sampling, we never reset the RNG state.
        """
        # Deliberately do nothing - we maintain a single continuous RNG
        pass
    
    # Method removed - no longer needed with continuous IID sampling
    
    def _get_file_handle(self):
        """Get file handle, creating it if needed (thread-local)"""
        if not hasattr(self.local, 'file') or self.local.file is None or self.local.file.closed:
            # Close any existing file handle first to prevent resource leaks
            self._close_file_handle()
            # Open a new file handle
            self.local.file = open(self.filepath, 'rb')
        return self.local.file
        
    def _close_file_handle(self):
        """Explicitly close file handle to free resources"""
        if hasattr(self.local, 'file') and self.local.file is not None and not self.local.file.closed:
            try:
                self.local.file.close()
                self.local.file = None
            except Exception as e:
                print(f"Error closing file handle: {e}")
    
    def __getitem__(self, idx):
        # For extremely large files, random.randint() can fail to provide proper distribution
        # Use direct multiplication approach which works better with large integer ranges
        file_pos = int(self.rng.random() * self.max_start)
        
        # Debug sampling
        if hasattr(self, 'sample_counter') and self.sample_counter % 1000 == 0 and hasattr(self, 'checkpoint_dir') and self.checkpoint_dir:
            sample_debug_path = os.path.join(self.checkpoint_dir, "random_debug.txt")
            with open(sample_debug_path, 'a' if os.path.exists(sample_debug_path) else 'w') as f:
                f.write(f"Sample {self.sample_counter}: position {file_pos:,} / {self.max_start:,}\n")
        
        # Increment counter for hash logging
        self.sample_counter += 1
        
        try:
            # Use memory mapping for faster access
            data = self.mmap[file_pos:file_pos+self.seq_len+1]  # +1 for target token
            
            # Convert to tensor
            tensor = torch.tensor(data, dtype=torch.long)
            
            # Handle edge case
            if tensor.size(0) < self.seq_len + 1:
                padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
                tensor = torch.cat([tensor, padding])
                
            # Calculate and log SHA256 hash of the sample if requested
            if self.log_sample_hashes and hasattr(self, 'sample_hash_file'):
                # Convert tensor to bytes and hash it
                sample_bytes = tensor.cpu().numpy().tobytes()
                sample_hash = hashlib.sha256(sample_bytes).hexdigest()
                
                # Write to hash log file with file position included
                with open(self.sample_hash_file, 'a') as f:
                    f.write(f"{self.sample_counter-1}\t{file_pos}\t{sample_hash}\n")
            
            return tensor
        except Exception as e:
            print(f"Error reading from file at position {file_pos}: {e}")
            # Try to recover file handle
            try:
                if hasattr(self.local, 'file') and self.local.file:
                    self.local.file.close()
                    self.local.file = None
            except:
                pass
            return torch.zeros(self.seq_len + 1, dtype=torch.long)
    
    def __del__(self):
        """Clean up resources - thread-local storage approach"""
        try:
            self._close_file_handle()
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
    
    # Disable torch.compile due to "incorrect arg count" errors with dynamo
    # if hasattr(torch, 'compile') and callable(torch.compile):
    #     model = torch.compile(model)
    #     print("Model compiled with torch.compile")
    print("torch.compile disabled to avoid dynamo errors")
    
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
    parser.add_argument('--master_addr', type=str, default=None,
                        help='Master node address (overrides environment variable)')
    
    # DeepSpeed arguments
    parser.add_argument('--deepspeed', action='store_true',
                       help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to DeepSpeed configuration file')
    parser.add_argument('--gradient_clipping', type=float, default=None,
                       help='Gradient clipping value for DeepSpeed')
    
    # Training arguments
    parser.add_argument('--train_steps', type=str, default="100",
                        help='number of training steps')
    parser.add_argument('--port', type=int, default=3442,
                        help='port for distributed communication')
    parser.add_argument('--data', type=str, required=True,
                        help='path to training data file')
    parser.add_argument('--val_mod', type=int, default=10,
                        help='modulo for validation set (1/10th for validation)')
    parser.add_argument('--output', type=str, default=None,
                        help='directory to save checkpoints (auto-generated if not specified)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    parser.add_argument('--validate_every', type=int, default=50,
                        help='validate and save checkpoint every N steps')
    parser.add_argument('--save_every', type=int, default=100,
                        help='additional checkpoints every N steps (optional, validation runs will always save)')
    parser.add_argument('--batches_per_epoch', type=str, default="100",
                        help='number of batches to include in each epoch (supports k/m/g suffixes)')
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
    
    # Add gradient accumulation parameter
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='number of gradient accumulation steps (effectively multiplies batch size)')
                        
    # Add sampling log parameters
    parser.add_argument('--log_sample_hashes', action='store_true',
                        help='log SHA256 hashes of samples for uniqueness validation')
    
    # Add a mutually exclusive group for backend selection
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument('--cuda', action='store_true', 
                        help='Use CUDA backend (for NVIDIA GPUs)')
    backend_group.add_argument('--rocm', action='store_true',
                        help='Use ROCm backend (for AMD GPUs)')
    
    # Add sf_beta2 parameter
    parser.add_argument('--sf_beta2', type=float, default=0.999,
                        help='ScheduleFree/Adam beta2 parameter (second moment decay)')
    
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
        
        # Print device health status only if there's an issue
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > current_device
        if not cuda_ok:
            print(f"WARNING: Rank {rank} has CUDA device issues. Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")
        
        # Staggered approach: small delay based on rank to avoid thundering herd
        time.sleep(0.01 * rank)
        
        # Simple barrier with device_ids to prevent hanging
        torch.distributed.barrier(device_ids=[current_device])
        
        # Sleep a bit after barrier to let things settle
        time.sleep(0.01)
            
    except Exception as e:
        print(f"ERROR in synchronize_processes: {e}")
        # Try to proceed anyway

def verify_gpu_health(use_rocm):
    """Verify that all GPUs are working properly"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if use_rocm:
            print(f"Found {device_count} AMD GPUs with ROCm/HIP")
        else:
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

# Environment setup is now handled by the batch script

def main():
    args = get_args()
    
    # Configure backend based on command-line arguments
    use_rocm = configure_backend(args)
    
    # Let local_rank come from DeepSpeed launcher via args
    local_rank = args.local_rank
    
    # Log distributed environment variables before initialization
    if local_rank <= 0:  # Could be -1 before proper initialization
        print("Environment variables seen by Python script before initialization:")
        for var in ["MASTER_ADDR", "MASTER_PORT", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            print(f"  {var}={os.environ.get(var, 'Not set')}")
    
    # Handle GPU exclusion if specified
    if args.exclude_gpus:
        exclude_list = [int(x.strip()) for x in args.exclude_gpus.split(',')]
        print(f"Excluding GPUs: {exclude_list}")
        
        # If current GPU is in exclude list, disable CUDA for this process
        if args.local_rank in exclude_list:
            print(f"Rank {args.local_rank} was specified to be excluded, forcing CPU execution")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # local_rank was already initialized at the beginning of the function
    # Will be updated after DeepSpeed init
    
    # Log environment variables for debugging
    if local_rank == 0:
        print("Environment variables for distributed training:")
        for var in ["MASTER_ADDR", "MASTER_PORT", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"]:
            print(f"{var}={os.environ.get(var, 'Not set')}")
    
    # Print debug information (initial)
    debug_distributed_info()
    
    # Verify GPU health before proceeding (only on rank 0)
    if local_rank == 0:
        verify_gpu_health(use_rocm)
    
    # Print memory usage monitoring message
    if local_rank <= 0:  # Use local_rank before DeepSpeed initialization
        print("Memory-efficient dataset initialized. Only loading necessary data.")
    
    # Parse numeric arguments with potential suffixes
    train_steps = int(parse_size_with_suffix(args.train_steps))
    seq_len = int(parse_size_with_suffix(args.seq_len))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    batches_per_epoch = int(parse_size_with_suffix(args.batches_per_epoch))
    
    # Update args with parsed integer values for DeepSpeed
    # Modify the args object to ensure integers are used
    args.batch_size = batch_size
    args.grad_accum = int(args.grad_accum)  # Ensure grad_accum is also an integer
    
    # Debug print to verify types
    print(f"DEBUG: batch_size type: {type(args.batch_size)}, value: {args.batch_size}")
    print(f"DEBUG: grad_accum type: {type(args.grad_accum)}, value: {args.grad_accum}")
    
    # Check for resuming from checkpoint
    resuming = args.resume is not None
    resume_step = 0
    loaded_config = None
    
    # Determine checkpoint_dir consistently across all ranks
    checkpoint_dir = None
    resuming = args.resume is not None

    if resuming:
        if os.path.isdir(args.resume):
            checkpoint_dir = args.resume
        else:
            checkpoint_dir = os.path.dirname(args.resume)
        if local_rank <= 0:  # Use local_rank before DeepSpeed initialization
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"Using checkpoint directory: {checkpoint_dir}")
            if not os.path.exists(checkpoint_dir):
                print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
                return
    elif args.output:
        checkpoint_dir = args.output
        if local_rank <= 0:  # Use local_rank before DeepSpeed initialization
            print(f"Using specified output directory: {checkpoint_dir}")
    else:
        # Create default name based on date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"gruboros_{timestamp}"
        if local_rank <= 0:  # Use local_rank before DeepSpeed initialization
            print(f"Output directory not specified, auto-generating: {checkpoint_dir}")

    # Rank 0 creates the directory if it doesn't exist (and isn't resuming)
    if local_rank <= 0 and checkpoint_dir and not resuming:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
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
        
        # Log gradient accumulation settings
        if args.grad_accum > 1:
            effective_batch = batch_size * args.grad_accum
            print(f"Using gradient accumulation: {args.grad_accum} steps")
            print(f"Effective batch size: {effective_batch} (micro-batch: {batch_size})")
        
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
            betas=(args.sf_beta, args.sf_beta2),
            weight_decay=args.weight_decay
        )
        if args.local_rank == 0:
            print(f"Using ScheduleFree optimizer with lr={args.lr}, beta1={args.sf_beta}, beta2={args.sf_beta2}")
    else:
        # Use standard AdamW (only when explicitly disabled)
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.sf_beta, args.sf_beta2),
            weight_decay=args.weight_decay
        )
        if args.local_rank == 0:
            print(f"Using standard AdamW optimizer with lr={args.lr}, beta1={args.sf_beta}, beta2={args.sf_beta2} (ScheduleFree disabled)")
    
    # Initialize DeepSpeed engine - let DeepSpeed handle distributed initialization
    print(f"Initializing DeepSpeed with {'ROCM' if use_rocm else 'CUDA'}")
    
    # Explicitly verify and fix batch size and grad_accum types right before DeepSpeed init
    if not isinstance(args.batch_size, int):
        print(f"WARNING: args.batch_size is not an int, fixing... Current type: {type(args.batch_size)}")
        args.batch_size = int(args.batch_size)
    
    if not isinstance(args.grad_accum, int):
        print(f"WARNING: args.grad_accum is not an int, fixing... Current type: {type(args.grad_accum)}")
        args.grad_accum = int(args.grad_accum)
    
    # Let DeepSpeed handle the distributed initialization
    # This is critical for the direct launcher to work correctly
    deepspeed_dist_init = True
    print(f"DeepSpeed dist_init_required set to: {deepspeed_dist_init} (letting DeepSpeed handle initialization)")
    
    # Explicitly set the distributed backend to gloo if NCCL isn't working
    # This is safer on systems without proper NCCL support
    if os.environ.get('TORCH_DISTRIBUTED_BACKEND', '').lower() == 'gloo':
        print("Using gloo backend for distributed training (NCCL disabled)")
        # Initialize distributed with gloo backend before DeepSpeed
        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(backend='gloo')
                print("Successfully initialized torch.distributed with gloo backend")
                # If we manually initialized, don't let DeepSpeed re-initialize
                deepspeed_dist_init = False
            except Exception as e:
                print(f"Warning: Failed to initialize gloo backend: {e}")
                print("Continuing with DeepSpeed default initialization")
    
    # Print world_info environment before DeepSpeed init
    print("\nDebug - environment variables for world_info:")
    for env_var in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK", 
                   "SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID"]:
        print(f"  {env_var}={os.environ.get(env_var, 'Not set')}")
        
    # If using a config file, make a direct modification to relevant DeepSpeed fields in-memory
    if args.deepspeed_config:
        # Load the config file to modify it in-memory
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        
        # Set batch sizes directly in the config to ensure they're integers
        if 'train_micro_batch_size_per_gpu' in ds_config:
            ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
        if 'gradient_accumulation_steps' in ds_config:
            ds_config['gradient_accumulation_steps'] = args.grad_accum
    
        # Set gradient clipping if provided as an argument
        if args.gradient_clipping is not None:
            ds_config['gradient_clipping'] = args.gradient_clipping
        
        # Print the config for debugging
        print(f"DeepSpeed config with batch sizes: {json.dumps(ds_config, indent=2)}")
        
        try:
            # Set the communication backend to gloo in the config if NCCL is disabled
            if os.environ.get('TORCH_DISTRIBUTED_BACKEND', '').lower() == 'gloo':
                ds_config['communication_backend'] = 'gloo'
                print("Setting DeepSpeed communication_backend to gloo")

            # ONLY use the config file approach - don't pass args for DeepSpeed config
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
                model_parameters=model.parameters(),
                dist_init_required=deepspeed_dist_init
            )
        except Exception as e:
            print(f"ERROR in DeepSpeed initialization with config (rank {local_rank}): {e}")
            # Try single-GPU fallback if distributed fails
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"Attempting fallback to single GPU training on device 0")
                try:
                    # Move model to first GPU
                    device = torch.device('cuda:0')
                    model = model.to(device)
                    
                    # Create a dummy model_engine object with minimal required attributes
                    class DummyModelEngine:
                        def __init__(self, model, optimizer, device):
                            self.module = model
                            self.optimizer = optimizer
                            self.device = device
                            self.local_rank = 0
                            self.global_rank = 0
                            self.world_size = 1
                    
                    model_engine = DummyModelEngine(model, optimizer, device)
                    print("Created fallback single-GPU engine")
                except Exception as fallback_error:
                    print(f"Single-GPU fallback also failed: {fallback_error}")
                    raise e
            else:
                raise
    else:
        # No config file, use args-only approach
        try:
            # Create a minimal config for gloo backend if needed
            if os.environ.get('TORCH_DISTRIBUTED_BACKEND', '').lower() == 'gloo':
                config_dict = {
                    "train_micro_batch_size_per_gpu": args.batch_size,
                    "gradient_accumulation_steps": args.grad_accum,
                    "communication_backend": "gloo"
                }
                model_engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    model_parameters=model.parameters(),
                    config=config_dict,
                    dist_init_required=deepspeed_dist_init
                )
            else:
                model_engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    model_parameters=model.parameters(),
                    config=None,
                    dist_init_required=deepspeed_dist_init
                )
        except Exception as e:
            print(f"ERROR in DeepSpeed initialization with args (rank {local_rank}): {e}")
            # Try single-GPU fallback if distributed fails
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"Attempting fallback to single GPU training on device 0")
                try:
                    # Move model to first GPU
                    device = torch.device('cuda:0')
                    model = model.to(device)
                    
                    # Create a dummy model_engine object with minimal required attributes
                    class DummyModelEngine:
                        def __init__(self, model, optimizer, device):
                            self.module = model
                            self.optimizer = optimizer
                            self.device = device
                            self.local_rank = 0
                            self.global_rank = 0
                            self.world_size = 1
                    
                    model_engine = DummyModelEngine(model, optimizer, device)
                    print("Created fallback single-GPU engine")
                except Exception as fallback_error:
                    print(f"Single-GPU fallback also failed: {fallback_error}")
                    raise e
            else:
                raise
        
    # Update args with ranks from model_engine after successful DeepSpeed initialization
    args.world_size = model_engine.world_size
    args.global_rank = model_engine.global_rank
    args.local_rank = model_engine.local_rank
    
    # Update local variable for easier access
    local_rank = args.local_rank

    # [CRITICAL FIX] Re-seed the Python random module for each process
    # to prevent synchronized mixing attempts. The other seeds (torch, numpy)
    # remain the same for deterministic weight initialization.
    random.seed(SEED + model_engine.global_rank)
    if model_engine.global_rank == 0:
        print(f"\nRe-seeded Python's random module per-rank to ensure stochastic mixing.\n")

    # Get data parallelism world size for token tracking
    dp_world_size = getattr(model_engine, 'data_parallel_world_size', 1)
    if not hasattr(model_engine, 'data_parallel_world_size'):
        # If attribute not available, try to calculate it
        tp_world_size = getattr(model_engine, 'tensor_parallel_world_size', args.tp_size)
        dp_world_size = max(1, args.world_size // tp_world_size)
    
    # Calculate data parallel rank (critical for correct TP+DP data loading)
    # With TP within node, this is effectively the node ID
    data_parallel_rank = model_engine.global_rank // args.tp_size

    # ===== EVOLUTIONARY GOSSIP INTEGRATION =====
    # Setup logging for gossip protocol
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir or ".", "gossip.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create evolutionary node with probability-based mixing
    evolutionary_node = EvolutionaryTrainingNode(
        node_id=f"node_{model_engine.global_rank}",
        model=model_engine.module,
        global_rank=model_engine.global_rank,
        world_size=model_engine.world_size,
        data_parallel_rank=data_parallel_rank,
        tp_size=args.tp_size,
        mixing_probability=0.01  # 1% chance to attempt a mix each step
    )
    
    # Start gossip protocol
    evolutionary_node.start_gossip_protocol()
    
    if model_engine.global_rank == 0:
        print("Evolutionary gossip protocol initialized")
        print(f"Node count: {model_engine.world_size}")
        print(f"Mixing probability: {evolutionary_node.mixing_probability * 100:.1f}% chance per step")
        print("Evolutionary strategy: Loser's weights are completely overwritten by winner's.")
    
    # Allow gossip protocol to initialize
    time.sleep(2)
    # ============================================
    
    # Create worker-specific seeds based on data parallel rank
    # This ensures all GPUs in the same TP group (node) get the same data
    worker_seed = SEED + data_parallel_rank
    print(f"Worker {model_engine.global_rank} (DP rank {data_parallel_rank}) using seed {worker_seed}")
    
    if model_engine.local_rank == 0:
        print(f"Data Parallel Rank: {data_parallel_rank}, DP World Size: {dp_world_size}")
        print(f"Tensor Parallel Size: {args.tp_size}")
    
    # Calculate batch sizes and tokens per step for accurate tracking
    micro_batch_per_gpu = batch_size  # Already parsed from args.batch_size
    grad_accum_steps = args.grad_accum
    effective_samples_per_gpu_update = micro_batch_per_gpu * grad_accum_steps
    effective_samples_per_node_update = effective_samples_per_gpu_update  # In this setup (TP within node)
    global_effective_samples_per_update = effective_samples_per_node_update * dp_world_size
    
    # Track actual tokens processed without gradient accumulation inflation
    global_samples_per_micro_batch = (micro_batch_per_gpu * dp_world_size)
    tokens_per_micro_batch_step = global_samples_per_micro_batch * seq_len
    
    print(f"DeepSpeed initialization successful: global_rank={args.global_rank}, "
          f"local_rank={args.local_rank}, world_size={args.world_size}")
    print(f"Parallelism: TP={args.tp_size}, DP={dp_world_size}, "
          f"Tokens per system step={tokens_per_micro_batch_step:,}")
    
    # Print updated debug info after DeepSpeed initialization
    if local_rank == 0:
        print("--- Distributed Info AFTER DeepSpeed Init ---")
        debug_distributed_info()
        print("---------------------------------------------")
    
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
        if hasattr(model_engine.optimizer, 'optimizer'):
            model_engine.optimizer.optimizer.train()
            print("Successfully set ScheduleFree optimizer to train mode via underlying optimizer.")
        else:
            # Try direct access approach
            try:
                model_engine.optimizer.train()
            except:
                print("Warning: Cannot access underlying optimizer for initial train mode in ScheduleFree.")
    
    # Calculate samples per epoch based on requested batches per epoch
    samples_per_epoch = batches_per_epoch * batch_size
    
    # Create the training dataset with continuous IID sampling
            
    train_dataset = ContinuousIIDDataset(
        filepath=args.data,
        seq_len=seq_len,
        seed=worker_seed,  # Use unique seed per worker
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        log_sample_hashes=args.log_sample_hashes,
        checkpoint_dir=checkpoint_dir if args.log_sample_hashes else None
    )
    
    # Create a separate validation dataset - use 10% of training batches
    val_batches = max(5, batches_per_epoch // 10)
    val_dataset = ContinuousIIDDataset(
        filepath=args.data,
        seq_len=seq_len,
        seed=worker_seed + 100,  # Validation seed still based on DP rank
        samples_per_epoch=val_batches * batch_size,
        batch_size=batch_size,
        log_sample_hashes=False  # Don't log validation samples
    )
    
    if args.local_rank == 0:
        print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Add seeding verification to confirm different data per rank
    if model_engine.global_rank == 0:
        print("=== SEEDING VERIFICATION ===")
        for rank in range(dp_world_size):
            rank_seed = SEED + rank
            temp_rng = random.Random(rank_seed)
            samples = [temp_rng.random() for _ in range(10)]
            print(f"DP Rank {rank} seed {rank_seed}: first 10 randoms: {samples[:5]}")
        print("========================")

    # Also verify the actual seeds being used:
    print(f"RANK {model_engine.global_rank}: worker_seed={worker_seed}, train_sampler seed used")
    
    # Create samplers for both datasets using UNIQUE SEEDS PER RANK
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dp_world_size,  # Data parallel world size
        rank=data_parallel_rank,     # Data parallel rank
        shuffle=True,
        seed=worker_seed  # ← FIXED: Use unique seed per rank
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dp_world_size,  # Data parallel world size
        rank=data_parallel_rank,     # Data parallel rank
        shuffle=False,
        seed=worker_seed + 1000  # ← FIXED: Unique val seed too
    )
    
    # Initialize datasets with epoch 0
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    
    # 6) DataLoaders with per-rank worker seeds
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # Increase slightly
        pin_memory=True,
        worker_init_fn=lambda wid: random.seed(worker_seed + wid + os.getpid()),  # ← Use worker_seed
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda wid: random.seed(worker_seed + 1000 + wid + os.getpid()),  # ← Unique val seed
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create metrics log file
    if model_engine.global_rank == 0 and checkpoint_dir:
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        with open(metrics_log_path, 'w') as f:
            header = [
                "step", "time_elapsed_s", "train_loss", "val_loss", 
                "total_tokens_processed_system", "tokens_per_sec_system", "learning_rate", "global_effective_batch_size_samples"
            ]
            f.write('\t'.join(header) + '\n')
    
    # Keep track of recent checkpoint files and what best points to
    recent_checkpoints = []
    best_checkpoint_filename = None
    
    # If resuming, check if there's already a best.pt file
    if resuming and model_engine.global_rank == 0 and checkpoint_dir:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path) or os.path.islink(best_path):
            # Try to get the filename it points to
            if os.path.islink(best_path):
                try:
                    target = os.readlink(best_path)
                    best_checkpoint_filename = os.path.basename(target)
                    print(f"Found existing best checkpoint: {best_checkpoint_filename}")
                except Exception as e:
                    print(f"Error reading existing best.pt symlink: {e}")
            else:
                print(f"best.pt exists as a regular file, not a symlink")

    # Helper function to save checkpoint - now with symlinks for both best and latest
    def save_checkpoint(step, train_loss, val_loss=None, is_best=False):
        if model_engine.global_rank != 0 or not checkpoint_dir:
            return
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
            
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
                except Exception as e:
                    print(f"Warning: Failed to remove previous best symlink: {e}")
            
            # Create symlink from best.pt to the current checkpoint with improved error handling
            try:
                # Make sure we're using absolute paths for reliable symlink creation
                abs_checkpoint_path = os.path.abspath(checkpoint_path)
                abs_best_path = os.path.abspath(best_path)
                
                # Print paths to help debug
                # Create the symlink (prefer relative paths for portability)
                os.symlink(os.path.basename(checkpoint_path), best_path)
                
                # Verify the symlink exists and resolve its target
                if not os.path.islink(best_path) or not os.path.exists(os.path.join(os.path.dirname(best_path), os.readlink(best_path))):
                    # Fallback to direct copy if symlink fails
                    os.remove(best_path) if os.path.exists(best_path) else None
                    shutil.copy2(checkpoint_path, best_path)
            except Exception as e:
                print(f"Warning: Failed to create best symlink, using copy instead")
                try:
                    # Ensure we have a best.pt file even if symlink fails
                    shutil.copy2(checkpoint_path, best_path)
                except Exception as copy_error:
                    print(f"Error: Could not create best.pt file: {copy_error}")
            
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
            if hasattr(model_engine.optimizer, 'optimizer'):
                model_engine.optimizer.optimizer.eval()
            else:
                # Try direct access approach
                try:
                    model_engine.optimizer.eval()
                except:
                    print("Warning: Cannot access underlying optimizer for eval mode in ScheduleFree.")
        
        # Don't display validation progress meter
        
        with torch.no_grad():
            # With continuous IID sampling, we no longer need to update epochs
            # val_sampler still needs epoch updates for proper distributed sampling
            val_sampler.set_epoch(current_epoch)
            
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
                    # Skip validation progress updates
                    pass
                
                # Limit validation to a few batches for speed
                if batch_count >= 5:
                    break
        
        # No validation bar to close
        
        # Calculate average
        avg_loss = total_loss / max(1, batch_count)
        
        # Synchronize all processes after validation to prevent some processes 
        # from continuing while others are still validating
        synchronize_processes()
        
        # Switch back to train mode
        model_engine.train()
        if args.schedulefree:
            if hasattr(model_engine.optimizer, 'optimizer'):
                model_engine.optimizer.optimizer.train()
            else:
                # Try direct access approach
                try:
                    model_engine.optimizer.train()
                except:
                    print("Warning: Cannot access underlying optimizer for train mode in ScheduleFree.")
        
        return avg_loss
    
    # Helper function to log metrics
    def log_metrics(step, train_loss, val_loss=None):
        if model_engine.global_rank != 0 or not checkpoint_dir:
            return
        
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        elapsed = time.time() - start_time
    
        # Calculate system-wide tokens processed and tokens per second
        total_tokens_processed_system = step * tokens_per_micro_batch_step
        tokens_per_sec_system = total_tokens_processed_system / elapsed if elapsed > 0 else 0
    
        current_lr = model_engine.optimizer.param_groups[0]['lr']
    
        values = [
            str(step),
            f"{elapsed:.2f}",
            f"{train_loss:.6f}",
            str(val_loss if val_loss is not None else "NA"),
            str(total_tokens_processed_system),
            f"{tokens_per_sec_system:.2f}",
            f"{current_lr:.8f}",
            str(global_effective_samples_per_update)
        ]
    
        with open(metrics_log_path, 'a') as f:
            f.write('\t'.join(values) + '\n')
    
    # 7) Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    current_val_loss = None  # Track most recent validation loss
    total_tokens_processed = 0  # Single-GPU tracking (legacy)
    total_tokens_processed_system = 0  # System-wide token tracking
    val_loss_for_checkpoint = None  # Initialize to None at start

    # If resuming, calculate tokens already processed
    if resuming:
        total_tokens_processed_system = resume_step * tokens_per_micro_batch_step
    
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
        print(f"- Data parallelism: {dp_world_size} nodes")
        print(f"- Keeping {args.keep_checkpoints} most recent checkpoints")
        print(f"- Gradient accumulation steps: {args.grad_accum}")
        print(f"- Micro-batch size: {batch_size} per GPU")
        print(f"- Effective batch size per GPU: {effective_samples_per_gpu_update}")
        print(f"- Effective batch size per node: {effective_samples_per_node_update}")
        print(f"- Global batch size: {global_effective_samples_per_update} samples ({tokens_per_micro_batch_step:,} tokens)")
        
        # Log the configuration summary
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "model_summary.txt"), "w") as f:
                f.write(f"Training command: {' '.join(sys.argv)}\n")
                f.write(f"Model parameters: {actual_params:,}\n")
                f.write(f"Configuration: {json.dumps(model_config, indent=2)}\n")
                f.write(f"Optimizer settings:\n")
                f.write(f"  Learning rate: {args.lr}\n")
                f.write(f"  Weight decay: {args.weight_decay}\n")
                f.write(f"  Beta1 (sf_beta): {args.sf_beta}\n")
                f.write(f"  Beta2 (sf_beta2): {args.sf_beta2}\n")
                f.write(f"  ScheduleFree enabled: {args.schedulefree}\n")
                f.write(f"Distributed setup:\n")
                f.write(f"  Tensor parallelism size (per node): {args.tp_size} GPUs\n")
                f.write(f"  Number of nodes (data parallel replicas): {dp_world_size}\n")
                f.write(f"  IMPORTANT: All GPUs within a node (TP group) receive identical data\n")
                f.write(f"  Sequence length: {seq_len}\n")
                f.write(f"  Micro-batch size per GPU: {micro_batch_per_gpu}\n")
                f.write(f"  Gradient accumulation steps: {grad_accum_steps}\n")
                f.write(f"  Effective batch size per GPU (after grad_accum): {effective_samples_per_gpu_update}\n")
                f.write(f"  Effective batch size per node (data parallel replica): {effective_samples_per_node_update}\n")
                f.write(f"  Global batch size (across all nodes, samples): {global_effective_samples_per_update}\n")
                f.write(f"  Global batch size (across all nodes, tokens): {tokens_per_micro_batch_step:,}\n")
    
    # Synchronize all processes before starting training loop
    synchronize_processes()
    
    # Create progress bar if primary process
    if model_engine.global_rank == 0:
        pbar = tqdm(total=train_steps, desc="Training", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        pbar.update(start_step)  # Update for resumed progress
        
    current_epoch = 0
    for step in range(start_step, start_step + train_steps):
        # Calculate current epoch based on batches_per_epoch parameter
        # With distributed training, we need to multiply by world_size since each GPU processes fewer batches
        effective_batches_per_epoch = batches_per_epoch
        epoch = step // effective_batches_per_epoch 
        
        # If we're starting a new epoch
        if epoch > current_epoch:
            current_epoch = epoch
            
            # Force garbage collection to clean up memory between epochs
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Explicitly reset ScheduleFree optimizer to train mode after epoch transition
            # This is critical as ScheduleFree uses two different points for gradient and test/val loss
            if args.schedulefree and hasattr(model_engine, 'optimizer'):
                if hasattr(model_engine.optimizer, 'optimizer'):
                    model_engine.optimizer.optimizer.train()
                else:
                    # Try direct access approach
                    try:
                        model_engine.optimizer.train()
                    except:
                        print("Warning: Cannot access underlying optimizer for train mode in ScheduleFree during epoch transition.")
        
        # Iterate through data loader for this step
        for batch_idx, x in enumerate(train_loader):
            # Split into input and target
            inputs, targets = x[:, :-1], x[:, 1:]
            
            # Move to device
            inputs, targets = inputs.to(model_engine.device), targets.to(model_engine.device)
            
            # Forward pass (returns loss directly)
            loss = model_engine(inputs, return_loss=True)
            
            # Get the loss value for logging. This is the unscaled loss for the micro-batch.
            loss_value = loss.detach().item()

            # --- DISABLE DEEPSPEED GRADIENT ALL-REDUCE ---
            # To enable our evolutionary gossip strategy, we must prevent DeepSpeed
            # from averaging gradients across all data-parallel ranks. We do this by
            # bypassing `model_engine.backward()` and calling the standard torch
            # `.backward()` on our own.
            
            # We must manually scale the loss for gradient accumulation, a task
            # that `model_engine.backward()` would normally handle.
            scaled_loss = loss / model_engine.gradient_accumulation_steps()
            
            # This computes gradients locally without any cross-GPU/node communication.
            # It correctly preserves the necessary all-reduce for Tensor Parallelism
            # while disabling the unwanted all-reduce for Data Parallelism.
            scaled_loss.backward()
            # ---------------------------------------------
            
            # ===== EVOLUTIONARY MIXING =====
            # Update evolutionary fitness WITH STEP NUMBER
            evolutionary_node.update_fitness(loss_value, step)
            
            # Check for incoming weight updates from other nodes
            update = evolutionary_node.check_for_updates()
            if update:
                evolutionary_node.apply_update(update)
                # Note: In practice, you might want to reset the optimizer state here
            
            # Let the node decide stochastically using its own per-rank RNG
            evolutionary_node.request_mix()
            
            # Log status periodically
            if step % 500 == 0 and model_engine.global_rank == 0:
                status = evolutionary_node.get_status()
                print(f"Evolutionary Status: {status}")
            # ===============================
            
            # Optimizer step (moved to AFTER mixing)
            model_engine.step()
            
            # Log progress
            if model_engine.global_rank == 0 and batch_idx == 0:
                # Update system-wide token tracking using our pre-calculated values
                total_tokens_processed_system += tokens_per_micro_batch_step
                
                # Calculate tokens per second across the whole system
                elapsed = time.time() - start_time
                tokens_per_sec_system = total_tokens_processed_system / elapsed if elapsed > 0 else 0
                
                # Update progress bar with stats in the format key=value
                formatted_stats = [
                    f"loss={loss.detach().item():.4f}",
                    f"lr={model_engine.optimizer.param_groups[0]['lr']:.6f}",
                    f"tok/s={tokens_per_sec_system:.2f}"
                ]
                
                # Add mixing statistics
                status = evolutionary_node.get_status()
                formatted_stats.append(f"mixes={status['mixing_attempts']},{status['successful_mixes']}")
                
                # Add validation info and epoch if available
                # Always show validation loss once we have calculated it at least once
                if current_val_loss is not None:
                    formatted_stats.append(f"val={current_val_loss:.4f}")
                    formatted_stats.append(f"best={best_val_loss:.4f}")
                
                # Add current epoch
                formatted_stats.append(f"epoch={current_epoch}")
                
                # Join all stats with commas for cleaner display
                postfix = " ".join(formatted_stats)
                
                pbar.set_postfix_str(postfix)
                pbar.update(1)
                
                # Log metrics for training
                log_metrics(step, loss.detach().item())
            
            # Break after one batch per step
            break
        
        # Validate periodically
        if args.validate_every > 0 and step > 0 and step % args.validate_every == 0:
            val_loss = validate()
            val_loss_for_checkpoint = val_loss
            current_val_loss = val_loss  # Store for display in progress bar
            
            if model_engine.global_rank == 0:
                # Log metrics with validation
                log_metrics(step, loss_value, val_loss)
            
            # Determine if this is the best model based on validation loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Always save checkpoint after validation, but silently
            save_path = save_checkpoint(step, loss_value, val_loss, is_best=is_best)
            
        
        # Save additional checkpoints periodically (if not already saved by validation)
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            # Skip if we just saved during validation
            if not (args.validate_every > 0 and step % args.validate_every == 0):
                # Save checkpoint without affecting best model status
                save_path = save_checkpoint(step, loss_value, None, is_best=False)
                
    
    # Final validation
    final_val_loss = validate()
    
    # Save final checkpoint and make it the best if it's better than previous best
    if model_engine.global_rank == 0:
        # Save the final checkpoint with validation loss and check if it's the best one
        is_final_best = final_val_loss < best_val_loss
        if is_final_best:
            best_val_loss = final_val_loss
        
        final_path = save_checkpoint(train_steps + start_step, loss_value, final_val_loss, is_best=is_final_best)
        
        # Ensure best.pt exists as final fallback
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if not os.path.exists(best_path) and not os.path.islink(best_path):
            shutil.copy2(final_path, best_path)
        
        # After training, switch to eval mode if using ScheduleFree
        if args.schedulefree:
            if hasattr(model_engine.optimizer, 'optimizer'):
                model_engine.optimizer.optimizer.eval()
            else:
                # Try direct access approach
                try:
                    model_engine.optimizer.eval()
                except:
                    print("Warning: Cannot access underlying optimizer for eval mode in ScheduleFree after training.")
        
        # Close progress bar
        pbar.close()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        tokens_per_sec_system = total_tokens_processed_system / total_time if total_time > 0 else 0
        
        # Print training summary
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Processed {total_tokens_processed_system:,} system-wide tokens")
        print(f"System throughput: {tokens_per_sec_system:.2f} tokens/sec")
        print(f"Final model saved to: {final_path}")
        
        # Cleanup evolutionary node
        evolutionary_node.stop_gossip_protocol()
        print("Evolutionary gossip protocol stopped")
        
        # Append final statistics to model summary
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "model_summary.txt"), "a") as f:
                f.write("\n----- Final Training Statistics -----\n")
                f.write(f"Total training time: {total_time:.2f} seconds\n")
                f.write(f"Total system tokens processed: {total_tokens_processed_system:,}\n")
                f.write(f"Average system tokens per second: {tokens_per_sec_system:.2f}\n")
                f.write(f"Final validation loss: {final_val_loss:.4f}\n")
                
                # Add evolutionary statistics
                final_status = evolutionary_node.get_status()
                f.write(f"\n----- Evolutionary Gossip Statistics -----\n")
                f.write(f"Final fitness: {final_status['fitness']:.4f}\n")
                f.write(f"Total mixing attempts: {final_status['mixing_attempts']}\n")
                f.write(f"Successful mixes: {final_status['successful_mixes']}\n")
                f.write(f"Mixing success rate: {final_status['successful_mixes']/max(1,final_status['mixing_attempts'])*100:.1f}%\n")
                f.write(f"Evolutionary strategy: Complete weight cloning (winner overwrites loser)\n")

if __name__ == "__main__":
    main()

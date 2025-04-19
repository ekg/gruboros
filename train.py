import os, random, numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import time
import argparse
import mmap
import re
from schedulefree import AdamWScheduleFree

# Import the minLM model
from mingru.minLM import minLM

# 1) Deterministic seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for efficient random access"""
    def __init__(self, filepath, seq_len):
        super().__init__()
        self.filepath = filepath
        self.seq_len = seq_len
        
        # Get file size
        self.file_size = os.path.getsize(filepath)
        
        # Calculate valid end position for random sampling
        # -seq_len-1 ensures we can always get seq_len+1 bytes
        self.valid_end = max(0, self.file_size - seq_len - 1)
        
        # Define length as number of possible starting positions
        self.length = max(1, self.valid_end)
        
        # Open file and create memory map - initialized on first access
        self.file = None
        self.mm = None
    
    def _ensure_open(self):
        """Ensure the memory map is open, opening it if necessary"""
        if self.mm is None:
            self.file = open(self.filepath, 'r+b')
            self.mm = mmap.mmap(self.file.fileno(), 0)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # This index is ignored - we generate a random position each time
        # This ensures randomness across different TP workers which need to see the same data
        self._ensure_open()
        
        # Generate a deterministic random position using seed + idx
        g = torch.Generator().manual_seed(SEED + idx)
        start_pos = torch.randint(0, self.valid_end, (1,), generator=g).item()
            
        # Directly read bytes from memory map without creating intermediate arrays
        self.mm.seek(start_pos)
        data = self.mm.read(self.seq_len + 1)  # +1 for the target
        
        # Convert to a writable buffer first, then to tensor
        writable_data = bytearray(data)
        tensor = torch.frombuffer(writable_data, dtype=torch.uint8).long()
        
        # Handle edge case if we didn't get enough data
        if tensor.size(0) < self.seq_len + 1:
            # Pad with zeros if needed
            padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        
        return tensor
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'mm') and self.mm is not None:
            self.mm.close()
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

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
    
    # DeepSpeed and distributed arguments
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallel size')
    
    # Training arguments
    parser.add_argument('--train_steps', type=str, default="100",
                        help='number of training steps (suffix: k=*1024, m=*1024*1024, g=*1024*1024*1024)')
    parser.add_argument('--port', type=int, default=29500,
                        help='port for distributed communication')
    parser.add_argument('--data', type=str, required=True,
                        help='path to training data file')
    
    # Model configuration
    parser.add_argument('--dim', type=str, default="512", 
                        help='model hidden dimension (suffix: k=*1024, m=*1024*1024, g=*1024*1024*1024)')
    parser.add_argument('--depth', type=int, default=6,
                        help='number of transformer layers')
    parser.add_argument('--seq_len', type=str, default="128",
                        help='sequence length for training (suffix: k=*1024, m=*1024*1024, g=*1024*1024*1024)')
    
    # Optimizer configuration
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--batch_size', type=str, default="4",
                        help='batch size per GPU (suffix: k=*1024, m=*1024*1024, g=*1024*1024*1024)')
    parser.add_argument('--schedulefree', action='store_true',
                        help='use ScheduleFree optimizer')
    parser.add_argument('--sf_beta', type=float, default=0.9,
                        help='ScheduleFree beta parameter')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set custom port for distributed communication
    if args.port != 29500:
        os.environ['MASTER_PORT'] = str(args.port)
    
    # Parse numeric arguments with potential suffixes
    train_steps = int(parse_size_with_suffix(args.train_steps))
    dim = int(parse_size_with_suffix(args.dim))
    seq_len = int(parse_size_with_suffix(args.seq_len))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    
    # Model configuration
    model_config = {
        "num_tokens": 256,  # byte-level tokenization
        "dim": dim,
        "depth": args.depth,
        "ff_mult": 4,
        "expansion": 1.5,
        "conv_kernel_size": 3,
        "use_lstm": False,
        "enable_conv": False,
        "dropout": 0.0
    }
    
    # Instantiate model
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
        print(f"Using ScheduleFree optimizer with lr={args.lr}, beta={args.sf_beta}")
    else:
        # Use standard AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"Using AdamW optimizer with lr={args.lr}")
    
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
    
    # 3) Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config_params=ds_config
    )
    
    # 4) Put ScheduleFree optimizer into train mode if used
    if args.schedulefree:
        model_engine.optimizer.train()
    
    # 5) Prepare Dataset & Sampler
    dataset = MemoryMappedDataset(
        filepath=args.data,
        seq_len=seq_len
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=model_engine.world_size,
        rank=model_engine.global_rank,
        shuffle=True,
        seed=SEED
    )
    
    # 6) DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda wid: torch.manual_seed(SEED + wid),
        persistent_workers=False  # Avoid persistent workers to prevent hanging
    )
    
    # 7) Training loop
    start_time = time.time()
    
    for step in range(train_steps):
        # Set epoch for deterministic shuffling
        epoch = step // len(data_loader)
        sampler.set_epoch(epoch)
        
        # Iterate through data loader for this step
        for batch_idx, x in enumerate(data_loader):
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
                elapsed = time.time() - start_time
                print(f"[Step {step:03d}] Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
            
            # Break after one batch per step
            break
    
    # After training, switch to eval mode if using ScheduleFree
    if args.schedulefree:
        model_engine.optimizer.eval()
    
    # Print training summary
    if model_engine.global_rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()

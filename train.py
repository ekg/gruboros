import os, random, numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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

# Import the minLM model and gossip protocol
from mingru.minLM import minLM
import logging
from gossip import EvolutionaryTrainingNode

# --- 1. SETUP AND CONFIGURATION ---

def configure_backend(args):
    """Configure environment for the selected backend (CUDA or ROCm)"""
    use_rocm = args.rocm
    if use_rocm and "SLURM_NODEID" in os.environ:
        os.environ["MIOPEN_USER_DB_PATH"] = f"/tmp/{os.environ.get('USER', 'user')}-miopen-cache-{os.environ['SLURM_NODEID']}"
        os.environ["MIOPEN_SYSTEM_DB_PATH"] = os.environ["MIOPEN_USER_DB_PATH"]
    if not use_rocm:
        torch.set_float32_matmul_precision('high')
    return use_rocm

def debug_distributed_info(global_rank):
    """Print debug information about the distributed environment"""
    print(f"\n----- Rank {global_rank} Distributed Environment Debug Info -----")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    if dist.is_initialized():
        print(f"Distributed is initialized: World={dist.get_world_size()}, Rank={dist.get_rank()}, Backend={dist.get_backend()}")
    else:
        print(f"Distributed is NOT initialized")
    print(f"------------------------------------------\n")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def round_to_multiple(n, multiple=64):
    return multiple * round(n / multiple)

def solve_for_dimension(target_params, depth, vocab_size=256, ff_mult=4, expansion=1.5):
    factor = 4 * expansion + 2 * ff_mult
    a = depth * factor
    b = 2 * vocab_size
    c = -target_params
    discriminant = b**2 - 4*a*c
    if discriminant < 0: raise ValueError("No solution exists")
    dim = (-b + math.sqrt(discriminant)) / (2*a)
    return round_to_multiple(dim)

def solve_for_depth(target_params, dim, vocab_size=256, ff_mult=4, expansion=1.5):
    embed_params = 2 * dim * vocab_size
    factor = 4 * expansion + 2 * ff_mult
    layer_params = dim * dim * factor
    depth = (target_params - embed_params) / layer_params
    return max(1, round(depth))

def calculate_model_size(config):
    dim, depth, vocab_size, ff_mult, expansion = config["dim"], config["depth"], config["num_tokens"], config["ff_mult"], config["expansion"]
    embedding_params = 2 * dim * vocab_size
    layer_params = dim * dim * (2 * expansion + 2 * ff_mult)
    total_params = embedding_params + depth * layer_params
    return int(total_params)

def get_parameter_count_str(config):
    params = calculate_model_size(config)
    if params >= 1e9: return f"{params/1e9:.1f}B"
    if params >= 1e6: return f"{params/1e6:.1f}M"
    return f"{params/1e3:.1f}K"

class ContinuousIIDDataset(Dataset):
    def __init__(self, filepath, seq_len, seed=42, samples_per_epoch=10000, batch_size=1, global_rank=0):
        super().__init__()
        self.filepath, self.seq_len, self.seed = filepath, seq_len, seed
        self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        self.max_start = len(self.mmap) - seq_len - 1
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(self.seed)
        if global_rank == 0:
            print(f"ContinuousIIDDataset: Using file {filepath} ({len(self.mmap):,} bytes)")
            print(f"Training with {samples_per_epoch // batch_size} batches per epoch")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        file_pos = int(self.rng.random() * self.max_start)
        data = self.mmap[file_pos:file_pos + self.seq_len + 1]
        tensor = torch.tensor(data, dtype=torch.long)
        if tensor.size(0) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        return tensor

def get_model(model_config):
    return minLM(**model_config)

def parse_size_with_suffix(size_str):
    if not isinstance(size_str, str): return size_str
    pattern = r'^(\d+(?:\.\d+)?)([kmg])?$'
    match = re.match(pattern, size_str.lower())
    if not match: return float(size_str)
    value, suffix = match.groups()
    value = float(value)
    if suffix == 'k': return value * 1024
    elif suffix == 'm': return value * 1024 * 1024
    elif suffix == 'g': return value * 1024 * 1024 * 1024
    return value

def get_args():
    parser = argparse.ArgumentParser(description='Pure Gossip Evolutionary Training for minLM')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank from launcher')
    parser.add_argument('--train_steps', type=str, default="100k", help='number of training steps')
    parser.add_argument('--data', type=str, required=True, help='path to training data file')
    parser.add_argument('--output', type=str, default=None, help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume')
    parser.add_argument('--validate_every', type=int, default=1000, help='save checkpoint based on EMA fitness every N steps')
    parser.add_argument('--save_every', type=int, default=2000, help='additional checkpoints every N steps')
    parser.add_argument('--batches_per_epoch', type=str, default="100", help='batches per epoch for dataloader length')
    parser.add_argument('--params', type=str, default="100m", help='target parameter count (e.g., 15m, 1g)')
    parser.add_argument('--dim', type=str, default=None, help='model hidden dimension (overrides params calculation)')
    parser.add_argument('--depth', type=int, default=None, help='number of layers (overrides params calculation)')
    parser.add_argument('--seq_len', type=str, default="2k", help='sequence length')
    parser.add_argument('--batch_size', type=str, default="4", help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='number of recent checkpoints to keep')
    parser.add_argument('--no-schedulefree', dest='schedulefree', action='store_false', default=True)
    parser.add_argument('--sf_beta', type=float, default=0.9)
    parser.add_argument('--sf_beta2', type=float, default=0.999)
    parser.add_argument('--gossip_merge_method', type=str, default='recombination', choices=['clonal', 'recombination'],
                        help='Method for merging models after gossip: clonal (overwrite) or recombination (mix).')
    parser.add_argument('--gossip_recombination_alpha', type=float, default=0.5,
                        help='Interpolation factor for recombination (0=loser, 1=winner).')
    parser.add_argument('--gossip_optimizer_recombination', type=str, default='interpolate', choices=['reset', 'interpolate'],
                        help='How to handle optimizer state during recombination: reset it or interpolate it.')
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument('--cuda', action='store_true')
    backend_group.add_argument('--rocm', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # --- 2. DISTRIBUTED INITIALIZATION (MANUAL) ---
    configure_backend(args)
    
    global_rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        dist.init_process_group(backend='gloo', timeout=timedelta(seconds=7200))
        print(f"Rank {global_rank}/{world_size} initialized process group with GLOO backend.")
    
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    random.seed(SEED + global_rank)
    if global_rank == 0:
        debug_distributed_info(global_rank)
        print(f"\nRe-seeded Python's random module per-rank to ensure stochastic mixing.\n")

    # --- 3. MODEL, OPTIMIZER, AND DATA SETUP ---
    train_steps = int(parse_size_with_suffix(args.train_steps))
    seq_len = int(parse_size_with_suffix(args.seq_len))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    batches_per_epoch = int(parse_size_with_suffix(args.batches_per_epoch))

    resuming = args.resume is not None
    resume_step = 0
    checkpoint_dir = args.output or f"gruboros_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if resuming:
        checkpoint_dir = os.path.dirname(args.resume) if os.path.isfile(args.resume) else args.resume

    if global_rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if world_size > 1: dist.barrier()

    model_config = None
    checkpoint = None
    if resuming:
        path = args.resume if os.path.isfile(args.resume) else os.path.join(args.resume, "latest.pt")
        if os.path.exists(path):
            if global_rank == 0: print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location='cpu')
            model_config, resume_step = checkpoint.get('model_config'), checkpoint.get('step', 0)

    if model_config is None:
        if args.dim and args.depth:
            dim = int(parse_size_with_suffix(args.dim))
            depth = args.depth
        else:
            params_value = parse_size_with_suffix(args.params)
            base_dim = 512 if params_value < 1e9 else 1024
            dim_guess = round_to_multiple(base_dim * (params_value / (100e6 if params_value < 1e9 else 1e9))**0.25)
            depth = solve_for_depth(params_value, dim_guess)
            dim = solve_for_dimension(params_value, depth)
        model_config = {"num_tokens": 256, "dim": dim, "depth": depth, "ff_mult": 4.0, "expansion": 1.5, "enable_conv": False, "dropout": 0.0}

    if global_rank == 0:
        print(f"Model size: {get_parameter_count_str(model_config)} parameters")
        print(f"Configuration: {model_config}")
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    model = get_model(model_config).to(device)

    optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay) if args.schedulefree else AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
    
    if resuming and checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if global_rank == 0: print(f"Resumed model and optimizer from step {resume_step}")
    
    if args.schedulefree: optimizer.train()

    train_dataset = ContinuousIIDDataset(args.data, seq_len, seed=SEED + global_rank, samples_per_epoch=batches_per_epoch * batch_size, batch_size=batch_size, global_rank=global_rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=SEED)
    
    # Calculate DataLoader workers based on hardware topology
    ranks_per_node = int(os.environ.get('RANKS_PER_NODE', 1))
    cpus_available = os.cpu_count() or 1
    cpus_per_rank = cpus_available // ranks_per_node
    num_workers = max(0, cpus_per_rank - 1)  # Reserve 1 CPU for main thread
    
    if num_workers > 0:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=4, 
            drop_last=True
        )
    else:
        # Fallback for single-CPU environments
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=True
        )
    
    if global_rank == 0:
        print(f"DataLoader configuration: {num_workers} workers per rank ({cpus_per_rank} CPUs per rank, {ranks_per_node} ranks per node)")
    
    # --- 4. GOSSIP AND TRAINING LOOP ---
    if global_rank == 0:
        # --- CENTRALIZED DIRECTORY CREATION ---
        # Rank 0 is responsible for creating ALL necessary subdirectories.
        print("Rank 0: Creating output subdirectories...")
        os.makedirs(os.path.join(checkpoint_dir, "gossip"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "metrics"), exist_ok=True)
        print("Rank 0: Directory creation complete.")

    # --- SYNCHRONIZATION BARRIER ---
    # All ranks wait here until rank 0 has finished creating the directories.
    # This prevents race conditions on the filesystem.
    if world_size > 1:
        print(f"Rank {global_rank}: Waiting at directory barrier...")
        dist.barrier()
        print(f"Rank {global_rank}: Passed directory barrier.")

    # Create per-rank metrics log file
    metrics_dir = os.path.join(checkpoint_dir, "metrics")
    metrics_log_path = os.path.join(metrics_dir, f"training_metrics_rank_{global_rank:03d}.tsv")
    with open(metrics_log_path, 'w') as f:
        header = [
            "rank", "step", "time_elapsed_s", "train_loss", "ema_fitness", 
            "total_tokens_processed", "tokens_per_sec", "learning_rate"
        ]
        f.write('\t'.join(header) + '\n')

    evolutionary_node = EvolutionaryTrainingNode(
        node_id=f"node_{global_rank}", model=model, optimizer=optimizer, global_rank=global_rank,
        local_rank=local_rank, world_size=world_size, data_parallel_rank=global_rank,
        tp_size=1, mixing_probability=0.01, output_dir=checkpoint_dir,
        merge_method=args.gossip_merge_method,
        recombination_alpha=args.gossip_recombination_alpha,
        optimizer_recombination=args.gossip_optimizer_recombination
    )
    evolutionary_node.start_gossip_protocol()
    if global_rank == 0: print("Evolutionary gossip protocol initialized and running.")

    start_time = time.time()
    best_ema_fitness = 0.0
    loss_value = 0.0
    total_tokens_processed = 0
    pbar = tqdm(total=train_steps, desc="Training", initial=resume_step, disable=(global_rank != 0))

    # Helper function to log metrics (per-rank)
    def log_metrics(step, train_loss, ema_fitness=None):
        nonlocal total_tokens_processed
        elapsed = time.time() - start_time
        total_tokens_processed = step * batch_size * seq_len
        tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']

        values = [
            str(global_rank),
            str(step),
            f"{elapsed:.2f}",
            f"{train_loss:.6f}",
            f"{ema_fitness:.6f}" if ema_fitness is not None else "NA",
            str(total_tokens_processed),
            f"{tokens_per_sec:.2f}",
            f"{current_lr:.8f}"
        ]

        with open(metrics_log_path, 'a') as f:
            f.write('\t'.join(values) + '\n')

    for step in range(resume_step, train_steps):
        train_sampler.set_epoch(step)
        
        # We only need one batch per step from the loader
        batch = next(iter(train_loader))

        # Apply pending gossip update before the step
        was_updated, needs_optimizer_reset = evolutionary_node.apply_pending_update()
        
        if needs_optimizer_reset:
            # Re-create the optimizer to reset its state
            if global_rank == 0:
                print(f"Rank {global_rank} resetting optimizer state at step {step} due to recombination.")
            optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay) if args.schedulefree else AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
            if args.schedulefree: optimizer.train()
            # Update the node's reference to the new optimizer
            evolutionary_node.optimizer = optimizer
        
        inputs, targets = batch[:, :-1].to(device, non_blocking=True), batch[:, 1:].to(device, non_blocking=True)
        
        # Note: Gradient accumulation is simplified here for clarity. 
        # A full implementation would loop `grad_accum` times before optimizer.step().
        optimizer.zero_grad()
        loss = model(inputs, return_loss=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.detach().item()
        
        # Update fitness and let gossip protocol run in the background
        evolutionary_node.update_fitness(loss_value, step)
        evolutionary_node.check_for_updates()
        evolutionary_node.request_mix()

        # Log metrics for ALL ranks
        current_ema_fitness = evolutionary_node.get_current_fitness()
        log_metrics(step, loss_value, current_ema_fitness)

        if global_rank == 0:
            status = evolutionary_node.get_status()
            pbar.set_postfix_str(f"loss={loss_value:.4f} ema_fit={status['fitness']:.4f} mixes={status['successful_mixes']}")
            pbar.update(1)

        # Rank 0 handles all checkpointing
        if global_rank == 0 and step > 0 and step % args.validate_every == 0:
            current_ema_fitness = evolutionary_node.get_current_fitness()
            is_best = current_ema_fitness > best_ema_fitness
            if is_best: best_ema_fitness = current_ema_fitness

            checkpoint_data = {'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'ema_fitness': current_ema_fitness, 'model_config': model_config}
            filename = f"step-{step:06d}-fit-{current_ema_fitness:.4f}.pt"
            filepath = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint_data, filepath)
            
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.lexists(latest_path): os.remove(latest_path)
            os.symlink(os.path.basename(filepath), latest_path)

            if is_best:
                best_path = os.path.join(checkpoint_dir, "best.pt")
                if os.path.lexists(best_path): os.remove(best_path)
                os.symlink(os.path.basename(filepath), best_path)

    pbar.close()
    evolutionary_node.stop_gossip_protocol()
    if global_rank == 0: print("\nTraining complete.")

if __name__ == "__main__":
    main()

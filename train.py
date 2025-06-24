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
import glob
from tqdm import tqdm
from schedulefree import AdamWScheduleFree
import fcntl
import contextlib
import threading
import atexit

# Import the minLM model and gossip protocol
from mingru.minLM import minLM
import logging
from gossip import EvolutionaryTrainingNode

class CheckpointManager:
    """Background thread for rank 0 to handle symlinks and cleanup"""
    
    def __init__(self, checkpoint_dir, check_interval=10, keep_last_n=5, keep_elite_n=10, global_rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.check_interval = check_interval
        self.keep_last_n = keep_last_n
        self.keep_elite_n = keep_elite_n
        self.global_rank = global_rank
        self.running = False
        self.thread = None
        self._stop_event = threading.Event()
        self.active = (global_rank == 0)
        # Regex to parse our new checkpoint filenames with loss
        self.ckpt_pattern = re.compile(r'checkpoint_rank_(\d+)_step_(\d+)_loss_([\d.inf]+)\.pt')
        
    def start(self):
        if not self.active or self.running:
            return
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._manager_loop, daemon=True)
        self.thread.start()
        atexit.register(self.stop)
        # if self.global_rank == 0:
        #     print("Started checkpoint manager thread")
        
    def stop(self):
        if not self.active or not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            
    def _manager_loop(self):
        while self.running and not self._stop_event.is_set():
            try:
                self._update_latest_symlink()
                self._update_best_symlink()
                self._update_elite_symlinks()
                self._cleanup_old_checkpoints()
                self._cleanup_tmp_files()
            except Exception as e:
                print(f"Rank 0: Checkpoint manager error: {e}")
            if self._stop_event.wait(timeout=self.check_interval):
                break
                
    def _parse_checkpoint(self, filepath):
        """Parses metadata from a checkpoint filename."""
        match = self.ckpt_pattern.search(os.path.basename(filepath))
        if not match:
            return None
        return {
            'path': filepath,
            'rank': int(match.group(1)),
            'step': int(match.group(2)),
            'loss': float(match.group(3))
        }

    def _atomic_symlink(self, target_basename, symlink_path):
        """Atomically create or update a symlink."""
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == target_basename:
            return
        temp_symlink = symlink_path + ".tmp"
        if os.path.lexists(temp_symlink):
            os.remove(temp_symlink)
        os.symlink(target_basename, temp_symlink)
        os.rename(temp_symlink, symlink_path)

    def _update_latest_symlink(self):
        try:
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt")
            checkpoint_files = glob.glob(pattern)
            if not checkpoint_files:
                return
            newest_file = max(checkpoint_files, key=os.path.getmtime)
            newest_basename = os.path.basename(newest_file)
            latest_symlink = os.path.join(self.checkpoint_dir, "latest.pt")
            self._atomic_symlink(newest_basename, latest_symlink)
        except Exception as e:
            print(f"Rank 0: Latest symlink update failed: {e}")

    def _update_best_symlink(self):
        try:
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt")
            checkpoint_files = glob.glob(pattern)
            if not checkpoint_files:
                return

            parsed_checkpoints = [p for p in (self._parse_checkpoint(f) for f in checkpoint_files) if p]
            if not parsed_checkpoints:
                return

            best_ckpt = min(parsed_checkpoints, key=lambda x: x['loss'])
            best_basename = os.path.basename(best_ckpt['path'])
            best_symlink = os.path.join(self.checkpoint_dir, "best.pt")
            self._atomic_symlink(best_basename, best_symlink)
        except Exception as e:
            print(f"Rank 0: Best symlink update failed: {e}")

    def _update_elite_symlinks(self):
        """Update ranked elite model symlinks (elite_01.pt, elite_02.pt, etc.)"""
        try:
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt")
            checkpoint_files = glob.glob(pattern)
            if not checkpoint_files:
                return

            parsed_checkpoints = [p for p in (self._parse_checkpoint(f) for f in checkpoint_files) if p]
            if not parsed_checkpoints:
                return

            # Sort by loss (ascending - lower is better) to get the elite models
            elite_checkpoints = sorted(parsed_checkpoints, key=lambda x: x['loss'])[:self.keep_elite_n]
            
            # Create/update elite symlinks
            for i, elite_ckpt in enumerate(elite_checkpoints, 1):
                elite_basename = os.path.basename(elite_ckpt['path'])
                elite_symlink = os.path.join(self.checkpoint_dir, f"elite_{i:02d}.pt")
                self._atomic_symlink(elite_basename, elite_symlink)
            
            # Remove any extra elite symlinks if we have fewer elite models than before
            for i in range(len(elite_checkpoints) + 1, self.keep_elite_n + 1):
                elite_symlink = os.path.join(self.checkpoint_dir, f"elite_{i:02d}.pt")
                if os.path.islink(elite_symlink):
                    os.remove(elite_symlink)
                    
        except Exception as e:
            print(f"Rank 0: Elite symlinks update failed: {e}")
            
    def _cleanup_old_checkpoints(self):
        try:
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt")
            checkpoint_files = glob.glob(pattern)
            if len(checkpoint_files) <= self.keep_last_n:
                return

            # Find the best checkpoint path to preserve it
            best_path_target = None
            best_symlink = os.path.join(self.checkpoint_dir, "best.pt")
            if os.path.islink(best_symlink):
                best_path_target = os.path.realpath(best_symlink)

            # Find all elite checkpoint paths to preserve them
            elite_path_targets = set()
            for i in range(1, self.keep_elite_n + 1):
                elite_symlink = os.path.join(self.checkpoint_dir, f"elite_{i:02d}.pt")
                if os.path.islink(elite_symlink):
                    elite_path_targets.add(os.path.realpath(elite_symlink))

            # Sort by modification time to find the N most recent to keep
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            
            # Identify files to keep: the N newest, plus the best one, plus all elite ones
            files_to_keep = set(checkpoint_files[:self.keep_last_n])
            if best_path_target and os.path.exists(best_path_target):
                files_to_keep.add(best_path_target)
            # Add all elite targets
            for elite_target in elite_path_targets:
                if os.path.exists(elite_target):
                    files_to_keep.add(elite_target)

            files_to_remove = [f for f in checkpoint_files if f not in files_to_keep]
            removed_count = 0
            for filepath in files_to_remove:
                try:
                    os.remove(filepath)
                    removed_count += 1
                except OSError:
                    continue
            # if removed_count > 0:
            #     print(f"Rank 0: Removed {removed_count} old checkpoints, preserved {len(files_to_keep)} "
            #           f"(including best and {len(elite_path_targets)} elite models).")
        except Exception as e:
            print(f"Rank 0: Checkpoint cleanup failed: {e}")
            
    def _cleanup_tmp_files(self):
        try:
            cutoff_time = time.time() - (10 * 60) # 10 minutes
            # Be more specific to only catch checkpoint temp files
            tmp_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt.tmp"))
            removed_count = 0
            for tmp_file in tmp_files:
                try:
                    if os.path.getmtime(tmp_file) < cutoff_time:
                        os.remove(tmp_file)
                        removed_count += 1
                except OSError:
                    continue
            # if removed_count > 0:
            #     print(f"Rank 0: Removed {removed_count} stale .tmp files")
        except Exception as e:
            print(f"Rank 0: Temp file cleanup failed: {e}")

def save_checkpoint_atomic(checkpoint_data, checkpoint_dir, step, global_rank, ema_fitness):
    """Save checkpoint atomically with loss in the filename."""
    filename = f"checkpoint_rank_{global_rank:04d}_step_{step:06d}_loss_{ema_fitness:.4f}.pt"
    temp_file = os.path.join(checkpoint_dir, filename + ".tmp")
    final_file = os.path.join(checkpoint_dir, filename)
    
    try:
        torch.save(checkpoint_data, temp_file)
        os.rename(temp_file, final_file)
        return True
    except Exception as e:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
        print(f"Rank {global_rank}: Checkpoint save failed: {e}")
        return False

@contextlib.contextmanager
def file_lock(lock_path, timeout=30):
    """Simple file-based lock with timeout"""
    lock_file = None
    try:
        lock_file = open(lock_path, 'w')
        # Try to acquire lock with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield lock_file
                return
            except BlockingIOError:
                time.sleep(0.1)
        raise TimeoutError(f"Could not acquire lock {lock_path} within {timeout}s")
    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except:
                pass


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

def _read_last_line(filepath):
    """Robustly reads the last non-empty line of a file."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0: return None
            f.seek(-2, os.SEEK_END)
            while f.tell() > 0 and f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode('utf-8').strip()
            if not last_line: # Handle files ending with multiple newlines
                f.seek(0)
                lines = f.readlines()
                for line in reversed(lines):
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line: return decoded_line
                return None
            return last_line
    except (IOError, OSError):
        return None


class ContinuousIIDDataset(Dataset):
    def __init__(self, filepath, chunk_size, context_chunks=1, seed=42, samples_per_epoch=10000, batch_size=1, global_rank=0):
        super().__init__()
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.context_chunks = context_chunks
        self.total_seq_len = self.chunk_size * self.context_chunks
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch

        self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        self.max_start = len(self.mmap) - self.total_seq_len - 1
        self.rng = random.Random(self.seed)

        if global_rank == 0:
            print(f"ContinuousIIDDataset: Using file {filepath} ({len(self.mmap):,} bytes)")
            print(f"Training with {samples_per_epoch // batch_size} meta-steps per epoch.")
            print(f"Effective sequence length: {self.total_seq_len} ({self.context_chunks} chunks of {self.chunk_size})")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        file_pos = int(self.rng.random() * self.max_start)
        # Fetch one long, contiguous sequence for TBPTT
        data = self.mmap[file_pos : file_pos + self.total_seq_len + 1]
        tensor = torch.tensor(data, dtype=torch.long)
        
        # This padding should rarely, if ever, be needed with correct max_start calculation
        if tensor.size(0) < self.total_seq_len + 1:
            padding = torch.zeros(self.total_seq_len + 1 - tensor.size(0), dtype=torch.long)
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
    parser.add_argument('--save_every', type=int, default=2000, help='Target average interval (in steps) for one checkpoint to be saved across the entire population.')
    parser.add_argument('--batches_per_epoch', type=str, default="100", help='batches per epoch for dataloader length')
    parser.add_argument('--params', type=str, default="100m", help='target parameter count (e.g., 15m, 1g)')
    parser.add_argument('--dim', type=str, default=None, help='model hidden dimension (overrides params calculation)')
    parser.add_argument('--depth', type=int, default=None, help='number of layers (overrides params calculation)')
    parser.add_argument('--chunk_size', type=str, default="2k", help='sequence length of each chunk for BPTT')
    parser.add_argument('--context_chunks', type=int, default=1, help='Number of consecutive chunks to process for TBPTT. Effective seq_len = chunk_size * context_chunks.')
    parser.add_argument('--batch_size', type=str, default="4", help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='number of recent checkpoints to keep')
    parser.add_argument('--keep_elite', type=int, default=10, help='number of elite models to preserve')
    parser.add_argument('--no-schedulefree', dest='schedulefree', action='store_false', default=True)
    parser.add_argument('--sf_beta', type=float, default=0.9)
    parser.add_argument('--sf_beta2', type=float, default=0.999)
    parser.add_argument('--gossip_merge_method', type=str, default='recombination', choices=['clonal', 'recombination'],
                        help='Method for merging models after gossip: clonal (overwrite) or recombination (mix).')
    parser.add_argument('--gossip_recombination_alpha', type=float, default=0.5,
                        help='Interpolation factor for recombination (0=loser, 1=winner).')
    parser.add_argument('--gossip_optimizer_recombination', type=str, default='interpolate', choices=['reset', 'interpolate'],
                        help='How to handle optimizer state during recombination: reset it or interpolate it.')
    parser.add_argument('--gossip_mixing_rate', type=float, default=0.01,
                        help='Probability of attempting evolutionary mixing each step (0.0-1.0).')
    parser.add_argument('--gossip_temp_dir', type=str, default=None,
                        help='Directory for temporary gossip payloads. Defaults to $SCRATCH or /tmp.')
    parser.add_argument('--gossip_fitness_decay', type=float, default=0.95,
                        help='Decay factor for the EMA loss used as fitness (e.g., 0.9 to 0.995).')
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument('--cuda', action='store_true')
    backend_group.add_argument('--rocm', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # --- 1. DISTRIBUTED INITIALIZATION ---
    configure_backend(args)
    
    global_rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        dist.init_process_group(backend='gloo', timeout=timedelta(seconds=7200))
    
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    random.seed(SEED + global_rank)

    if global_rank == 0:
        debug_distributed_info(global_rank)
        print(f"\nRe-seeded Python's random module per-rank to ensure stochastic mixing.\n")

    # --- 2. CONFIGURATION AND SETUP ---
    train_steps = int(parse_size_with_suffix(args.train_steps))
    chunk_size = int(parse_size_with_suffix(args.chunk_size))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    batches_per_epoch = int(parse_size_with_suffix(args.batches_per_epoch))
    
    resuming = args.resume is not None
    resume_step = 0
    checkpoint_dir = args.output or f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if resuming:
        checkpoint_dir = os.path.dirname(args.resume) if os.path.isfile(args.resume) else args.resume

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir, keep_last_n=args.keep_checkpoints,
        keep_elite_n=args.keep_elite, global_rank=global_rank
    )
    checkpoint_manager.start()

    model_config, checkpoint = None, None
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

        ### NEW: EXPLANATION OF THE TRAINING DYNAMICS ###
        print("\n--- Training Dynamics ---")
        print(f"A 'step' is one chunk of size {chunk_size}.")
        print(f"TBPTT context is carried for {args.context_chunks} chunks before reset.")
        print(f"Gradient accumulation steps: {args.grad_accum}")
        if args.grad_accum > 1:
            print(f"--> An optimizer step will occur every {args.grad_accum} chunk-steps.")
            print(f"--> Effective batch size (tokens): {batch_size * chunk_size * args.grad_accum}")
        else:
            print("--> An optimizer step will occur on every chunk-step.")
        print("-------------------------\n")


    model = get_model(model_config).to(device)
    optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay) if args.schedulefree else AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
    
    if resuming and checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if global_rank == 0: print(f"Resumed model and optimizer from step {resume_step}")
    
    if args.schedulefree: optimizer.train()

    train_dataset = ContinuousIIDDataset(
        args.data, chunk_size=chunk_size, context_chunks=args.context_chunks,
        seed=SEED + global_rank, samples_per_epoch=batches_per_epoch * batch_size,
        batch_size=batch_size, global_rank=global_rank
    )
    
    # Correct DataLoader setup from previous iteration
    ranks_per_node = int(os.environ.get('RANKS_PER_NODE', world_size if world_size > 0 else 1))
    cpus_available = os.cpu_count() or 1
    cpus_per_rank = (cpus_available // ranks_per_node) if ranks_per_node > 0 and ranks_per_node <= cpus_available else 1
    num_workers = max(0, cpus_per_rank - 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=None, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True
    )
    
    if global_rank == 0:
        print(f"DataLoader configuration: {num_workers} workers per rank ({cpus_per_rank} CPUs per rank, {ranks_per_node} ranks per node)")

    # --- 3. GOSSIP AND METRICS SETUP ---
    metrics_dir = os.path.join(checkpoint_dir, "metrics")
    metrics_log_path = os.path.join(metrics_dir, f"training_metrics_rank_{global_rank:03d}.tsv")
    with open(metrics_log_path, 'w') as f:
        f.write('\t'.join(["rank", "step", "time_elapsed_s", "train_loss", "ema_loss", "total_tokens_processed", "tokens_per_sec", "learning_rate"]) + '\n')

    evolutionary_node = EvolutionaryTrainingNode(
        node_id=f"node_{global_rank}", model=model, optimizer=optimizer, global_rank=global_rank,
        local_rank=local_rank, world_size=world_size, data_parallel_rank=global_rank,
        tp_size=1, mixing_probability=args.gossip_mixing_rate, output_dir=checkpoint_dir,
        merge_method=args.gossip_merge_method, recombination_alpha=args.gossip_recombination_alpha,
        optimizer_recombination=args.gossip_optimizer_recombination, gossip_temp_dir=args.gossip_temp_dir,
        fitness_decay_factor=args.gossip_fitness_decay
    )
    evolutionary_node.start_gossip_protocol()
    if global_rank == 0: print("Evolutionary gossip protocol initialized and running.")

    start_time = time.time()
    total_tokens_processed = 0
    save_probability = 1.0 / (args.save_every * world_size) if world_size > 1 and args.save_every > 0 else (1.0 / args.save_every if args.save_every > 0 else 0)

    # --- 4. UNIFIED TRAINING LOOP ---
    
    def log_metrics(step, train_loss, ema_loss=None):
        nonlocal total_tokens_processed
        elapsed = time.time() - start_time
        total_tokens_processed = (step + 1) * batch_size * chunk_size
        tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        values = [str(v) for v in [
            global_rank, step, f"{elapsed:.2f}", f"{train_loss:.6f}",
            f"{ema_loss:.6f}" if ema_loss is not None else "NA",
            total_tokens_processed, f"{tokens_per_sec:.2f}", f"{current_lr:.8f}"
        ]]
        with open(metrics_log_path, 'a') as f:
            f.write('\t'.join(values) + '\n')

    pbar = tqdm(total=train_steps, desc="Training", initial=resume_step, disable=(global_rank != 0))
    step = resume_step
    data_iterator = iter(train_loader)
    hidden_state = None
    optimizer.zero_grad() # Prepare for the first accumulation cycle.

    while step < train_steps:
        # Check for and apply any pending model updates at the beginning of the step.
        was_updated, needs_optimizer_reset = evolutionary_node.apply_pending_update()
        if needs_optimizer_reset:
            if global_rank == 0: print(f"Rank {global_rank} resetting optimizer state at step {step} due to recombination.")
            optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay) if args.schedulefree else AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
            if args.schedulefree: optimizer.train()
            evolutionary_node.optimizer = optimizer
            optimizer.zero_grad() # Crucially, zero the new optimizer's grads.

        # Fetch a new long batch if we're at the start of a context window.
        if step % args.context_chunks == 0:
            try:
                long_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                long_batch = next(data_iterator)
            # Reset hidden state for the new, unrelated sequence.
            hidden_state = None

        # Get the current chunk from the long batch.
        chunk_idx_in_batch = step % args.context_chunks
        start_idx = chunk_idx_in_batch * chunk_size
        chunk = long_batch[:, start_idx : start_idx + chunk_size + 1].to(device, non_blocking=True)
        
        # Forward pass for one chunk.
        loss, next_hidden_state = model(
            chunk, return_loss=True, return_prev_hiddens=True, prev_hiddens=hidden_state
        )
        
        chunk_loss = loss.detach().item()

        # Scale loss for gradient accumulation and perform backward pass.
        scaled_loss = loss / args.grad_accum
        scaled_loss.backward()

        # Detach hidden state for the next TBPTT step.
        if next_hidden_state:
            hidden_state = [h.detach() for h in next_hidden_state]
        
        # --- Optimizer Step Gate ---
        if (step + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # --- High-frequency operations follow (every chunk-step) ---
        evolutionary_node.update_fitness(chunk_loss, step)
        evolutionary_node.check_for_updates()
        evolutionary_node.request_mix()
        current_ema_fitness = evolutionary_node.get_current_fitness()
        log_metrics(step, chunk_loss, current_ema_fitness)

        if global_rank == 0:
            status = evolutionary_node.get_status()
            pbar.set_postfix_str(f"loss={chunk_loss:.4f} ema={status['fitness']:.4f} mixes={status['successful_mixes']}")
            pbar.update(1)

        # Probabilistic Saving
        if step > 0 and save_probability > 0 and random.random() < save_probability:
            checkpoint_data = {
                'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'ema_fitness': current_ema_fitness, 'model_config': model_config
            }
            save_checkpoint_atomic(checkpoint_data, checkpoint_dir, step, global_rank, current_ema_fitness)
        
        step += 1

    pbar.close()
    evolutionary_node.stop_gossip_protocol()
    checkpoint_manager.stop()
    if global_rank == 0: print("\nTraining complete.")

if __name__ == "__main__":
    main()

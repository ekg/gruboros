# CRITICAL: Set PATH before any torch imports for FlashRNN JIT compilation
# ninja must be available when torch.utils.cpp_extension is loaded
import os
os.environ['PATH'] = '/home/erikg/micromamba/envs/mingru/bin:' + os.environ.get('PATH', '')

import random, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
# from tqdm import tqdm  # Removed to save display space for gradient logging
from schedulefree import AdamWScheduleFree
import fcntl
import contextlib
import threading
import atexit

# Import the minLM model and gossip protocol
from mingru.minLM import minLM
import logging
from gossip import EvolutionaryTrainingNode
from data_utils import DocumentStreamDataset, SingleStreamDataset
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# === DETAILED PROFILING SETUP ===
import collections
phase_times_global = collections.defaultdict(list)
prof_step_times_global = {}  # {step: {phase: time}}

def prof_time():
    '''Get current time with CUDA sync'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
# === END PROFILING SETUP ===

def simple_barrier(barrier_name='default', timeout=300):
    """File-based barrier without MPI"""
    global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NPROCS', '1')))
    
    if world_size <= 1:
        return
    
    barrier_dir = os.path.join(os.environ.get('GOSSIP_TEMP_DIR', '/tmp'), 'barriers', barrier_name)
    os.makedirs(barrier_dir, exist_ok=True)
    barrier_file = os.path.join(barrier_dir, f'rank_{global_rank}.ready')
    
    # Signal ready
    Path(barrier_file).touch()
    
    # Wait for all ranks
    start_time = time.time()
    while len(glob.glob(os.path.join(barrier_dir, 'rank_*.ready'))) < world_size:
        if time.time() - start_time > timeout:
            print(f"Rank {global_rank}: Barrier timeout, continuing anyway...")
            break
        time.sleep(0.1)
    
    # Cleanup
    if global_rank == 0:
        time.sleep(0.5)  # Let others pass
        shutil.rmtree(barrier_dir, ignore_errors=True)

def setup_ddp_groups(global_rank, local_rank, world_size):
    """
    Setup DDP groups based on node topology.
    Returns: ddp_group, ddp_rank, ddp_world_size, is_ddp_primary, node_id
    """
    # Initialize process group if not already done
    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
    
    # Determine node ID (which node this rank is on)
    # This works for both SLURM and torchrun/deepspeed launchers
    if 'SLURM_NODEID' in os.environ:
        node_id = int(os.environ['SLURM_NODEID'])
        gpus_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', '8'))
    else:
        # For torchrun, ranks are assigned sequentially across nodes
        # Assume equal distribution
        gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
        node_id = global_rank // gpus_per_node
    
    # Find all ranks on the same node
    node_ranks = []
    for rank in range(world_size):
        # Check if this rank is on our node
        if 'SLURM_NODEID' in os.environ:
            # In SLURM, we can calculate directly
            rank_node = rank // gpus_per_node
        else:
            rank_node = rank // gpus_per_node
        
        if rank_node == node_id:
            node_ranks.append(rank)
    
    # Create DDP group for this node
    ddp_group = dist.new_group(node_ranks)
    
    # Determine position within DDP group
    ddp_rank = node_ranks.index(global_rank)
    ddp_world_size = len(node_ranks)
    
    # Only rank 0 within each DDP group participates in gossip
    is_ddp_primary = (ddp_rank == 0)
    
    return ddp_group, ddp_rank, ddp_world_size, is_ddp_primary, node_id

def update_symlinks_and_cleanup(checkpoint_dir, keep_last_n, keep_elite_n, milestone_every):
    """
    Simplified synchronous checkpoint management (NO background thread).
    Updates symlinks and cleans up old checkpoints immediately after saving.
    """
    ckpt_pattern = re.compile(r'checkpoint_rank_(\d+)_step_(\d+)_loss_([\d.]+)\.pt')

    # Get all checkpoint files
    all_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt"))

    # Parse checkpoint metadata
    parsed = []
    for filepath in all_files:
        match = ckpt_pattern.search(os.path.basename(filepath))
        if match:
            try:
                parsed.append({
                    'path': filepath,
                    'rank': int(match.group(1)),
                    'step': int(match.group(2)),
                    'loss': float(match.group(3)),
                    'mtime': os.path.getmtime(filepath)
                })
            except (FileNotFoundError, ValueError):
                continue

    if not parsed:
        return

    # Update latest.pt → newest checkpoint by mtime
    newest = max(parsed, key=lambda x: x['mtime'])
    latest_symlink = os.path.join(checkpoint_dir, "latest.pt")
    atomic_symlink(os.path.basename(newest['path']), latest_symlink)

    # Update best.pt → lowest loss checkpoint
    best = min(parsed, key=lambda x: x['loss'])
    best_symlink = os.path.join(checkpoint_dir, "best.pt")
    atomic_symlink(os.path.basename(best['path']), best_symlink)

    # Update elite_NN.pt → top N by loss
    elite_checkpoints = sorted(parsed, key=lambda x: x['loss'])[:keep_elite_n]
    for i, elite in enumerate(elite_checkpoints, 1):
        elite_symlink = os.path.join(checkpoint_dir, f"elite_{i:02d}.pt")
        atomic_symlink(os.path.basename(elite['path']), elite_symlink)

    # Update milestone_NNNNN.pt → checkpoints at milestone intervals
    milestone_checkpoints = []
    if milestone_every > 0:
        for ckpt in parsed:
            if ckpt['step'] % milestone_every == 0:
                milestone_checkpoints.append(ckpt)
        for ckpt in milestone_checkpoints:
            milestone_symlink = os.path.join(checkpoint_dir, f"milestone_{ckpt['step']:06d}.pt")
            atomic_symlink(os.path.basename(ckpt['path']), milestone_symlink)

    # Determine which files to keep
    elite_paths = {os.path.realpath(ckpt['path']) for ckpt in elite_checkpoints}
    recent_paths = {os.path.realpath(ckpt['path']) for ckpt in sorted(parsed, key=lambda x: x['mtime'], reverse=True)[:keep_last_n]}
    milestone_paths = {os.path.realpath(ckpt['path']) for ckpt in milestone_checkpoints}
    files_to_keep = elite_paths.union(recent_paths).union(milestone_paths)

    # Delete old checkpoints
    for ckpt in parsed:
        if os.path.realpath(ckpt['path']) not in files_to_keep:
            try:
                os.remove(ckpt['path'])
            except OSError:
                pass

def atomic_symlink(target_basename, symlink_path):
    """Atomically create or update a symlink."""
    if os.path.islink(symlink_path) and os.readlink(symlink_path) == target_basename:
        return
    temp_symlink = symlink_path + ".tmp"
    if os.path.lexists(temp_symlink):
        os.remove(temp_symlink)
    os.symlink(target_basename, temp_symlink)
    os.rename(temp_symlink, symlink_path)

class CheckpointManager:
    """Background thread for rank 0 to handle symlinks and cleanup"""
    
    def __init__(self, checkpoint_dir, check_interval=10, keep_last_n=5, keep_elite_n=10, global_rank=0, archive_rate=0.0, milestone_every=0):
        self.checkpoint_dir = checkpoint_dir
        self.check_interval = check_interval
        self.keep_last_n = keep_last_n
        self.keep_elite_n = keep_elite_n
        self.global_rank = global_rank
        self.archive_rate = archive_rate
        self.milestone_every = milestone_every
        self.archive_counter = 0
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
        """
        Main loop for the manager. It takes a snapshot of the directory,
        then performs all actions based on that consistent snapshot.
        """
        while self.running and not self._stop_event.is_set():
            try:
                # --- 1. TAKE THE SNAPSHOT ---
                # Get a single, consistent list of checkpoint files at this moment.
                pattern = os.path.join(self.checkpoint_dir, "checkpoint_rank_*_step_*_loss_*.pt")
                all_checkpoint_paths = glob.glob(pattern)
                
                # Parse all files from the snapshot. This list is now the "source of truth" for this cycle.
                parsed_checkpoints = [p for p in (self._parse_checkpoint(f) for f in all_checkpoint_paths) if p]

                # --- 2. ACT ON THE SNAPSHOT ---
                if parsed_checkpoints:
                    self._update_latest_symlink(parsed_checkpoints)
                    self._update_best_symlink(parsed_checkpoints)
                    self._update_elite_symlinks(parsed_checkpoints)
                    self._update_milestone_symlinks(parsed_checkpoints)
                    self._cleanup_old_checkpoints(parsed_checkpoints, all_checkpoint_paths)
                
                # Temp file cleanup can still run independently.
                self._cleanup_tmp_files()
                
            except Exception as e:
                print(f"Rank 0: Checkpoint manager error: {e}")

            if self._stop_event.wait(timeout=self.check_interval):
                break
                
    def _parse_checkpoint(self, filepath):
        """Parses metadata from a checkpoint filename, returning None if file vanishes."""
        match = self.ckpt_pattern.search(os.path.basename(filepath))
        if not match:
            return None
        try:
            # Also get mtime for sorting "latest" files
            mtime = os.path.getmtime(filepath)
            return {
                'path': filepath,
                'rank': int(match.group(1)),
                'step': int(match.group(2)),
                'loss': float(match.group(3)),
                'mtime': mtime
            }
        except FileNotFoundError:
            # The file was deleted between glob() and getmtime(), which is fine. Ignore it.
            return None

    def _atomic_symlink(self, target_basename, symlink_path):
        """Atomically create or update a symlink."""
        if os.path.islink(symlink_path) and os.readlink(symlink_path) == target_basename:
            return
        temp_symlink = symlink_path + ".tmp"
        if os.path.lexists(temp_symlink):
            os.remove(temp_symlink)
        os.symlink(target_basename, temp_symlink)
        os.rename(temp_symlink, symlink_path)

    def _has_archive_symlink(self, filepath):
        """Check if any archive symlink points to this file."""
        try:
            target_path = os.path.realpath(filepath)
            archive_pattern = os.path.join(self.checkpoint_dir, "archive_*.pt")
            archive_symlinks = glob.glob(archive_pattern)
            for symlink in archive_symlinks:
                if os.path.islink(symlink) and os.path.realpath(symlink) == target_path:
                    return True
            return False
        except Exception:
            return False

    def _update_latest_symlink(self, parsed_checkpoints):
        try:
            newest_file = max(parsed_checkpoints, key=lambda x: x['mtime'])
            newest_basename = os.path.basename(newest_file['path'])
            latest_symlink = os.path.join(self.checkpoint_dir, "latest.pt")
            self._atomic_symlink(newest_basename, latest_symlink)
        except Exception as e:
            print(f"Rank 0: Latest symlink update failed: {e}")

    def _update_best_symlink(self, parsed_checkpoints):
        try:
            best_ckpt = min(parsed_checkpoints, key=lambda x: x['loss'])
            best_basename = os.path.basename(best_ckpt['path'])
            best_symlink = os.path.join(self.checkpoint_dir, "best.pt")
            self._atomic_symlink(best_basename, best_symlink)
        except Exception as e:
            print(f"Rank 0: Best symlink update failed: {e}")

    def _update_elite_symlinks(self, parsed_checkpoints):
        try:
            elite_checkpoints = sorted(parsed_checkpoints, key=lambda x: x['loss'])[:self.keep_elite_n]

            for i, elite_ckpt in enumerate(elite_checkpoints, 1):
                elite_basename = os.path.basename(elite_ckpt['path'])
                elite_symlink = os.path.join(self.checkpoint_dir, f"elite_{i:02d}.pt")
                self._atomic_symlink(elite_basename, elite_symlink)

            # Remove any extra elite symlinks if we have fewer elite models than before
            # Check a wider range to be safe in case of manual deletions
            for i in range(len(elite_checkpoints) + 1, self.keep_elite_n + 20):
                elite_symlink = os.path.join(self.checkpoint_dir, f"elite_{i:02d}.pt")
                if os.path.islink(elite_symlink):
                    os.remove(elite_symlink)

        except Exception as e:
            print(f"Rank 0: Elite symlinks update failed: {e}")

    def _update_milestone_symlinks(self, parsed_checkpoints):
        """Create permanent milestone symlinks for checkpoints at milestone steps"""
        if self.milestone_every <= 0:
            return

        try:
            # Find all checkpoints that should have milestone symlinks
            milestone_checkpoints = [
                ckpt for ckpt in parsed_checkpoints
                if ckpt['step'] % self.milestone_every == 0
            ]

            # Create milestone symlink for each milestone checkpoint
            for milestone_ckpt in milestone_checkpoints:
                milestone_basename = os.path.basename(milestone_ckpt['path'])
                milestone_symlink = os.path.join(
                    self.checkpoint_dir,
                    f"milestone_{milestone_ckpt['step']:06d}.pt"
                )
                # Only create if it doesn't exist yet
                if not os.path.exists(milestone_symlink):
                    self._atomic_symlink(milestone_basename, milestone_symlink)

        except Exception as e:
            print(f"Rank 0: Milestone symlinks update failed: {e}")

    def _cleanup_old_checkpoints(self, parsed_checkpoints, all_checkpoint_paths):
        """Clean up based on the consistent snapshot."""
        # A simple check to avoid work if there's nothing to clean up.
        if len(all_checkpoint_paths) <= self.keep_last_n and len(all_checkpoint_paths) <= self.keep_elite_n:
            return
        
        try:
            # 1. Identify elite files to keep from our consistent list
            elite_checkpoints = sorted(parsed_checkpoints, key=lambda x: x['loss'])[:self.keep_elite_n]
            elite_paths = {os.path.realpath(ckpt['path']) for ckpt in elite_checkpoints}

            # 2. Identify the N most recent files to keep, also from the consistent list
            sorted_by_time = sorted(parsed_checkpoints, key=lambda x: x['mtime'], reverse=True)
            recent_paths = {os.path.realpath(ckpt['path']) for ckpt in sorted_by_time[:self.keep_last_n]}
            
            # 3. Identify files that are *already* protected by an archive symlink.
            archive_symlinks = glob.glob(os.path.join(self.checkpoint_dir, "archive_*.pt"))
            # Get the real, absolute path of the target file for each archive symlink.
            archived_target_paths = {os.path.realpath(s) for s in archive_symlinks if os.path.islink(s)}

            # 3b. Identify files that are protected by milestone symlinks.
            milestone_symlinks = glob.glob(os.path.join(self.checkpoint_dir, "milestone_*.pt"))
            # Get the real, absolute path of the target file for each milestone symlink.
            milestone_target_paths = {os.path.realpath(s) for s in milestone_symlinks if os.path.islink(s)}

            # 4. Combine ALL sets of files to preserve: elites, recents, archives, AND milestones.
            files_to_keep = elite_paths.union(recent_paths).union(archived_target_paths).union(milestone_target_paths)

            # 5. Determine which files to remove. This list will now correctly
            #    exclude any checkpoint that is already archived.
            all_paths_set = {os.path.realpath(f) for f in all_checkpoint_paths}
            files_to_remove = [f for f in all_checkpoint_paths if os.path.realpath(f) not in files_to_keep]
            
            for filepath in files_to_remove:
                try:
                    # This file is guaranteed not to have an archive link yet.
                    if self.archive_rate > 0 and random.random() < self.archive_rate:
                        self.archive_counter += 1
                        archive_basename = os.path.basename(filepath)
                        archive_symlink = os.path.join(self.checkpoint_dir, f"archive_{self.archive_counter:03d}.pt")
                        self._atomic_symlink(archive_basename, archive_symlink)
                    
                    # We still need this check, because we might have *just* created an archive link
                    # in the lines above.
                    if not self._has_archive_symlink(filepath):
                        os.remove(filepath)
                except OSError: # Catches FileNotFoundError and other issues
                    continue
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

def save_checkpoint_atomic(checkpoint_data, checkpoint_dir, step, global_rank, validation_fitness):
    """Save checkpoint atomically with loss in the filename."""
    filename = f"checkpoint_rank_{global_rank:04d}_step_{step:06d}_loss_{validation_fitness:.4f}.pt"
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

def save_checkpoint_background(checkpoint_data_cpu, checkpoint_dir, step, global_rank, validation_fitness, cleanup_fn=None):
    """
    Background thread function to save checkpoint to disk.
    checkpoint_data_cpu should already be on CPU.
    """
    success = save_checkpoint_atomic(checkpoint_data_cpu, checkpoint_dir, step, global_rank, validation_fitness)

    if success:
        if cleanup_fn:
            try:
                cleanup_fn()
                print(f"Rank 0: Checkpoint saved successfully (background)")
            except Exception as e:
                print(f"Rank 0: Symlink/cleanup failed: {e}")
    else:
        print(f"Rank 0: WARNING - Background checkpoint save failed!")

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


SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def round_to_multiple(n, multiple=64):
    return multiple * round(n / multiple)

def solve_for_dimension(target_params, depth, vocab_size=256, ff_mult=4, expansion=1.5, use_hybrid_gru=False):
    """Approximates the model dimension `d` for a target parameter count."""
    if use_hybrid_gru:
        # For HybridFusedGRU, iterate to find the right dimension
        # Start with an initial guess
        dim_guess = 512 if target_params < 1e9 else 1024
        
        for _ in range(20):  # Iterate to refine
            dim_inner = int(dim_guess * expansion)
            
            # Calculate params with current guess
            embed_params = 2 * dim_guess * vocab_size
            # Only include to_out projection if expansion != 1.0
            to_out_params = dim_inner * dim_guess if expansion != 1.0 else 0
            gru_params = (
                dim_guess * 3 * dim_inner + 3 * dim_inner +
                dim_inner * 3 * dim_inner + 3 * dim_inner +
                to_out_params
            )
            ffn_params = 2 * dim_guess * dim_guess * ff_mult if ff_mult > 0 else 0
            norm_params = depth * 2 * dim_guess + dim_guess
            
            total_params = embed_params + depth * (gru_params + ffn_params) + norm_params
            
            # Adjust guess based on ratio
            ratio = target_params / total_params
            dim_guess = int(dim_guess * math.sqrt(ratio))  # sqrt because params scale with dim^2
            
            if abs(total_params - target_params) / target_params < 0.01:  # Within 1%
                break
        
        return round_to_multiple(dim_guess)
    else:
        # Original minGRU formula
        factor = 3 * expansion + 2 * ff_mult
        a = depth * factor
        b = 2 * vocab_size
        c = -target_params
        discriminant = b**2 - 4*a*c
        if discriminant < 0: raise ValueError("No real solution for dimension exists with these parameters.")
        dim = (-b + math.sqrt(discriminant)) / (2*a)
        return round_to_multiple(dim)

def solve_for_depth(target_params, dim, vocab_size=256, ff_mult=4, expansion=1.5, use_hybrid_gru=False):
    """Approximates the model depth for a target parameter count."""
    # Calculate embedding params
    embed_params = 2 * dim * vocab_size
    
    # Calculate params per layer based on architecture
    if use_hybrid_gru:
        # HybridFusedGRU has full GRU architecture
        dim_inner = int(dim * expansion)
        # Only include to_out projection if expansion != 1.0
        to_out_params = dim_inner * dim if expansion != 1.0 else 0
        gru_params = (
            dim * 3 * dim_inner + 3 * dim_inner +  # input_projection with bias
            dim_inner * 3 * dim_inner + 3 * dim_inner +  # hidden_projection with bias
            to_out_params  # to_out without bias (only if expansion != 1.0)
        )
    else:
        # minGRU approximation
        gru_params = dim * dim * 3 * expansion
    
    # FFN and norm params
    ffn_params = 2 * dim * dim * ff_mult if ff_mult > 0 else 0
    norm_params = 2 * dim  # 2 layer norms per layer
    
    layer_params = gru_params + ffn_params + norm_params
    if layer_params <= 0: return 1
    
    # Account for final norm
    available_for_layers = target_params - embed_params - dim
    depth = available_for_layers / layer_params
    return max(1, round(depth))

def calculate_model_size(config):
    """Calculates the approximate parameter count of a minLM model."""
    dim = config["dim"]
    depth = config["depth"]
    vocab_size = config["num_tokens"]
    ff_mult = config.get("ff_mult", 4)
    expansion = config.get("expansion", 1.5)
    use_hybrid_gru = config.get("use_hybrid_gru", False)
    
    # Embeddings and output projection
    embedding_params = dim * vocab_size  # Token embeddings
    output_params = dim * vocab_size  # Output projection
    
    if use_hybrid_gru:
        # HybridFusedGRU has full GRU architecture
        dim_inner = int(dim * expansion)
        
        # Per GRU layer:
        # - input_projection: dim × (3 × dim_inner) + bias
        # - hidden_projection: dim_inner × (3 × dim_inner) + bias  
        # - to_out: dim_inner × dim (no bias)
        gru_params_per_layer = (
            dim * 3 * dim_inner + 3 * dim_inner +  # input_projection with bias
            dim_inner * 3 * dim_inner + 3 * dim_inner +  # hidden_projection with bias
            dim_inner * dim  # to_out without bias
        )
    else:
        # minGRU approximation
        gru_params_per_layer = dim * dim * 3 * expansion
    
    # FFN parameters (same for both)
    if ff_mult > 0:
        ffn_params_per_layer = 2 * dim * dim * ff_mult  # Two linear layers
    else:
        ffn_params_per_layer = 0
    
    # Layer norms: 2 per layer (before GRU and before FFN) 
    norm_params_per_layer = 2 * dim
    
    # Total per layer
    params_per_layer = gru_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    
    # Final layer norm
    final_norm_params = dim
    
    total_params = embedding_params + output_params + depth * params_per_layer + final_norm_params
    return int(total_params)

def get_parameter_count_str(config):
    params = calculate_model_size(config)
    if params >= 1e9: return f"{params/1e9:.2f}B"
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
    """Legacy dataset - replaced by DocumentStreamDataset for document-aware training"""
    def __init__(self, filepath, chunk_size, context_chunks=1, seed=42, samples_per_epoch=10000, batch_size=1, global_rank=0):
        super().__init__()
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.context_chunks = context_chunks
        self.total_seq_len = self.chunk_size * self.context_chunks
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch

        self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        self.max_start = len(self.mmap) - self.total_seq_len
        self.rng = random.Random(self.seed)

        if global_rank == 0:
            print(f"ContinuousIIDDataset: Using file {filepath} ({len(self.mmap):,} bytes)")
            print(f"Training with {samples_per_epoch} samples per epoch.")
            print(f"Effective sequence length: {self.total_seq_len} ({self.context_chunks} chunks of {self.chunk_size})")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        file_pos = int(self.rng.random() * self.max_start)
        # Fetch one long, contiguous sequence for TBPTT
        data = self.mmap[file_pos : file_pos + self.total_seq_len]
        tensor = torch.tensor(data, dtype=torch.long)
        
        # This padding should rarely, if ever, be needed with correct max_start calculation
        if tensor.size(0) < self.total_seq_len:
            padding = torch.zeros(self.total_seq_len - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        return tensor


# DocumentStreamDataset moved to data_utils.py
'''
class DocumentStreamDataset(Dataset):
    """
    Document-aware streaming dataset for training.
    
    Key features:
    - Respects document boundaries (0x1e delimiter)
    - Each GPU starts at different random position
    - Resets model hidden state at document boundaries
    - Tracks per-GPU statistics (not global)
    - Enables dynamic optimization at document ends
    """
    def __init__(self, filepath, chunk_size, seed=42, rank=None, global_rank=0, world_size=None, shared_mmap=None, tokenizer=None):
        super().__init__()
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        # Use shared_mmap if provided, otherwise create new memmap
        if shared_mmap is not None:
            self.mmap = shared_mmap
        else:
            self.mmap = np.memmap(filepath, dtype=np.uint8, mode='r')
        self.file_size = len(self.mmap)

        # Use rank if provided, otherwise use global_rank
        effective_rank = rank if rank is not None else global_rank

        # Each stream gets a unique starting position (seed already includes rank+stream offset from wrapper)
        rng = random.Random(seed)
        self.position = rng.randint(0, self.file_size - 1)
        
        # Per-GPU statistics (initialize before calling _scan_to_next_document)
        self.documents_processed = 0
        self.bytes_processed = 0  # File bytes read
        self.tokens_processed = 0  # Actual tokens generated (for TikToken, etc.)
        self.wraps = 0
        
        # Scan forward to next document boundary to start clean
        self._scan_to_next_document()

        # Buffer for accumulating tokens/bytes until we have a full chunk
        self.byte_buffer = []
        self.token_buffer = []  # For tokenized data

        # For tokenization: buffer text before tokenizing
        self.text_buffer = b''
        self.read_chunk_size = 32768  # Read 32KB at a time for tokenization (faster, fewer encode calls)

        print(f"Rank {effective_rank}: DocumentStreamDataset initialized at position {self.position}")
        if self.tokenizer is not None:
            print(f"Rank {effective_rank}: Using tokenizer: {self.tokenizer.__class__.__name__}")
        
    def _scan_to_next_document(self):
        """Scan forward to the start of the next document"""
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1
        
        if self.position >= self.file_size:
            self.position = 0
            self.wraps += 1
        else:
            self.position += 1  # Skip the \x1e delimiter
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1
    
    def get_next_chunk(self):
        """
        Returns: (chunk_tensor, is_final_chunk_in_doc, actual_chunk_length)

        IMPORTANT: Always returns fixed-size tensors for CUDA graph compatibility.
        Partial chunks are padded with zeros, and actual_length indicates valid data.
        """
        # Use byte-level logic if no tokenizer or byte tokenizer
        if self.tokenizer is None or self.tokenizer.__class__.__name__ == 'ByteTokenizer':
            return self._get_next_chunk_bytes()
        else:
            return self._get_next_chunk_tokens()

    def _get_next_chunk_bytes(self):
        """Original byte-level streaming logic."""
        while len(self.byte_buffer) < self.chunk_size:
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1

            byte_val = int(self.mmap[self.position])
            self.position += 1
            self.bytes_processed += 1

            if byte_val == 0x1e:
                self.documents_processed += 1

                if len(self.byte_buffer) > 0:
                    actual_length = len(self.byte_buffer)
                    chunk = torch.zeros(self.chunk_size, dtype=torch.long)
                    chunk[:actual_length] = torch.tensor(self.byte_buffer, dtype=torch.long)
                    self.byte_buffer = []
                    self.tokens_processed += actual_length  # For bytes: 1 byte = 1 token
                    return chunk, True, actual_length
                else:
                    continue
            else:
                self.byte_buffer.append(byte_val)

        chunk = torch.tensor(self.byte_buffer[:self.chunk_size], dtype=torch.long)
        self.byte_buffer = self.byte_buffer[self.chunk_size:]
        self.tokens_processed += self.chunk_size  # For bytes: 1 byte = 1 token
        return chunk, False, self.chunk_size

    def _get_next_chunk_tokens(self):
        """Token-level streaming with proper tokenization."""
        # DEBUG: First call only
        if not hasattr(self, '_token_debug_done'):
            import sys
            print(f"[DEBUG] _get_next_chunk_tokens called! Tokenizer: {self.tokenizer.__class__.__name__}", flush=True)
            sys.stdout.flush()
            self._token_debug_done = True

        # Refill token buffer if running low
        while len(self.token_buffer) < self.chunk_size:
            # Read a chunk of bytes
            bytes_to_read = min(self.read_chunk_size, self.file_size - self.position)
            if bytes_to_read == 0:
                # Wrap around
                self.position = 0
                self.wraps += 1
                bytes_to_read = min(self.read_chunk_size, self.file_size)

            byte_chunk = bytes(self.mmap[self.position:self.position + bytes_to_read])
            self.position += bytes_to_read
            self.bytes_processed += bytes_to_read

            # Check for document boundary (0x1e)
            doc_boundary_idx = byte_chunk.find(b'\x1e')

            if doc_boundary_idx != -1:
                # Found document boundary
                self.text_buffer += byte_chunk[:doc_boundary_idx]

                # Tokenize accumulated text if any
                if len(self.text_buffer) > 0:
                    try:
                        text = self.text_buffer.decode('utf-8', errors='ignore')
                        tokens = self.tokenizer.encode(text)
                        self.token_buffer.extend(tokens)
                        self.text_buffer = b''
                    except Exception as e:
                        print(f"Warning: tokenization error: {e}")
                        self.text_buffer = b''

                self.documents_processed += 1

                # Skip past delimiter and continue
                self.position = self.position - len(byte_chunk) + doc_boundary_idx + 1
                if self.position >= self.file_size:
                    self.position = 0
                    self.wraps += 1

                # Return partial chunk if we have tokens
                if len(self.token_buffer) > 0:
                    actual_length = min(len(self.token_buffer), self.chunk_size)
                    chunk = torch.zeros(self.chunk_size, dtype=torch.long)
                    chunk[:actual_length] = torch.tensor(self.token_buffer[:actual_length], dtype=torch.long)
                    self.token_buffer = self.token_buffer[actual_length:]
                    self.tokens_processed += actual_length  # Track tokens returned
                    return chunk, True, actual_length
                # Otherwise continue to next document
            else:
                # No boundary, accumulate text
                self.text_buffer += byte_chunk

                # Periodically tokenize to avoid huge text buffer
                if len(self.text_buffer) >= self.read_chunk_size * 4:
                    try:
                        text = self.text_buffer.decode('utf-8', errors='ignore')
                        tokens = self.tokenizer.encode(text)
                        self.token_buffer.extend(tokens)
                        self.text_buffer = b''
                    except Exception as e:
                        print(f"Warning: tokenization error: {e}")
                        self.text_buffer = b''

        # Return full chunk
        chunk = torch.tensor(self.token_buffer[:self.chunk_size], dtype=torch.long)
        self.token_buffer = self.token_buffer[self.chunk_size:]
        self.tokens_processed += self.chunk_size  # Track tokens returned
        return chunk, False, self.chunk_size
    
    def get_stats(self):
        return {
            'documents_processed': self.documents_processed,
            'bytes_processed': self.bytes_processed,
            'tokens_processed': self.tokens_processed,  # Actual tokens generated
            'file_wraps': self.wraps,
            'current_position': self.position
        }
'''

class DocumentStreamWrapper(IterableDataset):
    """
    Wrapper to make DocumentStreamDataset work with PyTorch DataLoader
    --- MODIFIED FOR BATCHING WITH SHARED MEMORY MAP ---
    """
    def __init__(self, filepath, chunk_size, batch_size, seed=42, global_rank=0, tokenizer=None):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.seed = seed
        self.global_rank = global_rank
        self.tokenizer = tokenizer

        # Create a single shared memory map for this rank
        import mmap
        self.data_file = open(filepath, 'rb')
        self.shared_mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)

        # Streams will be created lazily in __iter__ to incorporate worker_id
        self.streams = None
        
    def __iter__(self):
        # Lazily create streams to incorporate worker_id into seeding
        if self.streams is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            num_workers = worker_info.num_workers if worker_info is not None else 1

            # Each worker gets unique seeds: incorporate worker_id and num_workers
            base_seed = self.seed + (self.global_rank * 1000) + (worker_id * 100)

            self.streams = [
                DocumentStreamDataset(
                    self.filepath,
                    self.chunk_size,
                    rank=self.global_rank,
                    world_size=1,
                    seed=base_seed + i,
                    shared_mmap=self.shared_mmap,
                    tokenizer=self.tokenizer
                ) for i in range(self.batch_size)
            ]

        while True:
            batch_chunks, batch_is_doc_end, batch_actual_len = [], [], []

            for stream in self.streams:
                chunk, is_end, length = stream.get_next_chunk()
                batch_chunks.append(chunk)
                batch_is_doc_end.append(is_end)
                batch_actual_len.append(length)

            # Stack individual tensors into a single batch tensor
            yield (
                torch.stack(batch_chunks),
                torch.tensor(batch_is_doc_end, dtype=torch.bool),
                torch.tensor(batch_actual_len, dtype=torch.long)
            )
    
    def get_all_stats(self):
        """Get stats from all streams (returns empty list if streams not created yet)"""
        if self.streams is None:
            return []
        return [stream.get_stats() for stream in self.streams]

    def __del__(self):
        """Clean up the shared memory map and file handle"""
        if hasattr(self, 'shared_mmap'):
            self.shared_mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()

def get_model(model_config):
    # Use Mamba SSM if requested (gold standard RNN)
    if model_config.get('use_mamba', False):
        from mingru.mamba_lm import MambaLM
        mamba_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'd_state': model_config.get('mamba_d_state', 16),
            'expand': model_config.get('mamba_expand', 2),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using MambaLM: dim={mamba_config['dim']}, depth={mamba_config['depth']}, d_state={mamba_config['d_state']}, expand={mamba_config['expand']}")
        return MambaLM(**mamba_config)

    # Use Mamba2 SSM with SSD if requested (faster parallel training)
    if model_config.get('use_mamba2', False):
        from mingru.mamba_lm import Mamba2LM
        mamba2_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'd_state': model_config.get('mamba_d_state', 64),  # Mamba2 uses larger state by default
            'expand': model_config.get('mamba_expand', 2),
            'headdim': 64,  # Standard Mamba2 headdim
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using Mamba2LM: dim={mamba2_config['dim']}, depth={mamba2_config['depth']}, d_state={mamba2_config['d_state']}, expand={mamba2_config['expand']}")
        return Mamba2LM(**mamba2_config)

    # Use Hybrid Mamba2 + cuDNN GRU with parallel paths and learned mixing
    if model_config.get('use_hybrid_mamba2_gru', False):
        from mingru.hybrid_mamba2_gru import HybridMamba2GRULM
        hybrid_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'mamba_d_state': model_config.get('mamba_d_state', 64),
            'mamba_expand': model_config.get('mamba_expand', 2),
            'mamba_headdim': 64,
            'gru_expansion': 1.0,  # Match dim for fair comparison
            'ff_mult': 0.0,  # No FFN to match Mamba2 structure
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using HybridMamba2GRULM: dim={hybrid_config['dim']}, depth={hybrid_config['depth']}, d_state={hybrid_config['mamba_d_state']}, expand={hybrid_config['mamba_expand']}")
        return HybridMamba2GRULM(**hybrid_config)

    # Use Mamba2 + 3-layer FFN (based on arXiv:2505.06633)
    if model_config.get('use_mamba2_ffn3', False):
        from mingru.mamba2_ffn3 import Mamba2FFN3LM
        ffn3_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'mamba_d_state': model_config.get('mamba_d_state', 64),
            'mamba_expand': model_config.get('mamba_expand', 2),
            'mamba_headdim': 64,
            'ff_expansion': 4,  # 3-layer FFN: d → 4d → 4d → d
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using Mamba2FFN3LM: dim={ffn3_config['dim']}, depth={ffn3_config['depth']}, d_state={ffn3_config['mamba_d_state']}, ff_expansion={ffn3_config['ff_expansion']}")
        return Mamba2FFN3LM(**ffn3_config)

    # Use cuDNN GRU + Conv1d + Multiplicative Gating (Mamba2-style local context)
    if model_config.get('use_cudnn_conv_gru', False):
        from mingru.cudnn_gru_conv import CuDNNGRU_ConvLM
        conv_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'gate_expansion': 1.0,
            'conv_kernel': 4,  # Like Mamba2
            'use_input_gate': True,
            'use_hidden_gate': True,
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using CuDNNGRU_ConvLM: dim={conv_config['dim']}, depth={conv_config['depth']}, conv_kernel={conv_config['conv_kernel']}")
        return CuDNNGRU_ConvLM(**conv_config)

    # Use minGRU + SiLU multiplicative output gate (parallel scan + selectivity)
    if model_config.get('use_mingru_mult', False):
        from mingru.mingru_mult import minGRU_MultLM
        mingru_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'use_output_gate': True,  # SiLU output selectivity
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using minGRU_MultLM: dim={mingru_config['dim']}, depth={mingru_config['depth']}, expansion={mingru_config['expansion_factor']}")
        return minGRU_MultLM(**mingru_config)

    # Use cuDNN LSTM + SiLU multiplicative output gate (more stepwise nonlinearity)
    if model_config.get('use_cudnn_lstm_mult', False):
        from mingru.cudnn_lstm_mult import CuDNNLSTM_MultLM
        lstm_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'gate_activation': 'silu',  # Mamba2-style
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using CuDNNLSTM_MultLM: dim={lstm_config['dim']}, depth={lstm_config['depth']}, expansion={lstm_config['expansion_factor']}")
        return CuDNNLSTM_MultLM(**lstm_config)

    # Use Elman MLP + SiLU selectivity (simplest possible recurrence)
    if model_config.get('use_elman_selective', False):
        from mingru.elman_selective import ElmanSelectiveLM
        elman_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion': model_config['expansion'],
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using ElmanSelectiveLM: dim={elman_config['dim']}, depth={elman_config['depth']}, expansion={elman_config['expansion']}")
        return ElmanSelectiveLM(**elman_config)

    # Use cuDNN GRU + Multiplicative Gating (for testing nonlinearity hypothesis)
    if model_config.get('use_cudnn_mult_gru', False):
        from mingru.cudnn_gru_mult import CuDNNGRU_MultLM
        mult_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'gate_expansion': 1.0,  # Gate matches hidden dim
            'use_input_gate': True,  # Gate depends on input x
            'use_hidden_gate': not model_config.get('input_only_gate', False),  # Gate depends on hidden h (disabled for ablation)
            'use_glu_gate': model_config.get('use_glu_gate', False),  # GLU-style gate (h-dependent only)
            'use_swiglu': model_config.get('use_swiglu', False),  # SwiGLU-style output
            'gate_activation': model_config.get('gate_activation', 'sigmoid'),  # sigmoid or silu (like Mamba2)
            'ff_mult': model_config.get('ff_mult', 0.0),  # Optional FFN
            'dropout': model_config['dropout'],
            'tie_weights': True,
            'use_checkpointing': model_config.get('use_checkpointing', False),
            'inner_chunk_size': model_config.get('inner_chunk_size', 128),
        }
        ckpt_str = f", checkpointing={mult_config['inner_chunk_size']}" if mult_config['use_checkpointing'] else ""
        gate_str = f", gate={mult_config['gate_activation']}" if mult_config['gate_activation'] != 'sigmoid' else ""
        swiglu_str = ", swiglu" if mult_config['use_swiglu'] else ""
        print(f"Using CuDNNGRU_MultLM: dim={mult_config['dim']}, depth={mult_config['depth']}, ff_mult={mult_config['ff_mult']}{gate_str}{swiglu_str}{ckpt_str}")
        return CuDNNGRU_MultLM(**mult_config)

    # Use Haste GRU + SiLU Multiplicative Gating (fused CUDA kernel, BF16 native)
    # EXACT numerical equivalence to CuDNNGRU_MultLM with gate_activation='silu'
    if model_config.get('use_haste_mult_gru', False):
        from mingru.haste_gru_mult import HasteGRU_MultLM
        mult_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
            'use_checkpointing': model_config.get('use_checkpointing', False),
            'inner_chunk_size': model_config.get('inner_chunk_size', 128),
        }
        ckpt_str = f", checkpointing={mult_config['inner_chunk_size']}" if mult_config['use_checkpointing'] else ""
        print(f"Using HasteGRU_MultLM (fused CUDA, BF16 native): dim={mult_config['dim']}, depth={mult_config['depth']}, ff_mult={mult_config['ff_mult']}{ckpt_str}")
        return HasteGRU_MultLM(**mult_config)

    # Use Discretized Elman with explicit delta (leaky integrator)
    # Fast version using haste ElmanNoGate + EMA blending
    if model_config.get('use_discretized_elman', False):
        from mingru.discretized_elman_fast import DiscretizedElmanFastLM
        delta_mode = model_config.get('delta_mode', 'fixed')
        # Input-dependent delta REQUIRES exact mode (fast mode has broken dynamics)
        use_exact = delta_mode in ('input', 'input_state')
        elman_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'delta_mode': delta_mode,
            'delta_init': model_config.get('delta_init', 0.5),
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
            'add_output_gate': True,  # Mamba2-style input-only silu gate
            'exact': use_exact,  # True for input-dependent delta (correct dynamics)
        }
        print(f"Using DiscretizedElmanFastLM: dim={elman_config['dim']}, depth={elman_config['depth']}, delta_mode={elman_config['delta_mode']}, delta_init={elman_config['delta_init']}, exact={use_exact}")
        return DiscretizedElmanFastLM(**elman_config)

    # Use cuDNN GRU + Bilinear (true second-order h×x interactions)
    if model_config.get('use_cudnn_bilinear_gru', False):
        from mingru.cudnn_gru_bilinear import CuDNNGRU_BilinearLM
        bilinear_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'bilinear_rank': model_config['dim'],  # Full rank bilinear
            'use_gated_bilinear': True,  # Gate the bilinear term
            'ff_mult': model_config.get('ff_mult', 0.0),  # Optional FFN
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using CuDNNGRU_BilinearLM: dim={bilinear_config['dim']}, depth={bilinear_config['depth']}, bilinear_rank={bilinear_config['bilinear_rank']}")
        return CuDNNGRU_BilinearLM(**bilinear_config)

    # Use plain cuDNN GRU (no modifications - pure baseline)
    if model_config.get('use_cudnn_plain_gru', False):
        from mingru.cudnn_gru_plain import PlainCuDNNGRU_LM
        plain_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using PlainCuDNNGRU_LM: dim={plain_config['dim']}, depth={plain_config['depth']} (no modifications, pure GRU baseline)")
        return PlainCuDNNGRU_LM(**plain_config)

    # Use cuDNN GRU + 3-layer FFN Gate (deep selectivity mechanism)
    if model_config.get('use_cudnn_ffn3_gru', False):
        from mingru.cudnn_gru_ffn3 import CuDNNGRU_FFN3_LM
        ffn3_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion_factor': model_config['expansion'],
            'ffn_expand': model_config.get('ffn_expand', 2.0),  # FFN hidden dim multiplier
            'ff_mult': model_config.get('ff_mult', 0.0),
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using CuDNNGRU_FFN3_LM: dim={ffn3_config['dim']}, depth={ffn3_config['depth']}, ffn_expand={ffn3_config['ffn_expand']} (3-layer FFN selectivity gate on concat[x,h])")
        return CuDNNGRU_FFN3_LM(**ffn3_config)

    # Use EMA + Input Gate (Option 2: minimal architecture testing selectivity hypothesis)
    if model_config.get('use_ema_input_gate', False):
        from mingru.ema_input_gate import EMAInputGateLM
        ema_gate_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'ema_alpha': model_config.get('ema_alpha', 0.01),
            'ff_mult': model_config.get('ff_mult', 4.0),  # Need FFN for params
            'dropout': model_config['dropout'],
            'tie_weights': True,
        }
        print(f"Using EMAInputGateLM: dim={ema_gate_config['dim']}, depth={ema_gate_config['depth']}, ff_mult={ema_gate_config['ff_mult']} (EMA + input-dependent gate)")
        return EMAInputGateLM(**ema_gate_config)

    # Use deep normal-space GRU model if requested (RECOMMENDED!)
    if model_config.get('use_deep_normal_gru', False):
        from mingru.deep_normal_gru_lm import DeepNormalGRULM
        deep_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion': model_config['expansion'],
            'dropout': model_config['dropout'],
            'recurrence_chunk_size': model_config.get('recurrence_chunk_size', 64)
        }
        return DeepNormalGRULM(**deep_config)

    # Use log-space deep GRU model if requested (experimental, unstable)
    if model_config.get('use_logspace_gru', False):
        from mingru.deep_logspace_lm import DeepLogSpaceGRULM
        logspace_config = {
            'num_tokens': model_config['num_tokens'],
            'dim': model_config['dim'],
            'depth': model_config['depth'],
            'expansion': model_config['expansion'],
            'dropout': model_config['dropout'],
            'z_bias_input': model_config.get('z_bias_input', -2.0),
            'z_bias_hidden': model_config.get('z_bias_hidden', -2.0),
            'recurrence_chunk_size': model_config.get('recurrence_chunk_size', 64)
        }
        return DeepLogSpaceGRULM(**logspace_config)

    # Only pass keys that minLM actually accepts (whitelist approach)
    minlm_valid_keys = {
        'num_tokens', 'dim', 'depth', 'ff_mult', 'expansion', 'conv_kernel_size',
        'dropout', 'use_fused_gru', 'use_hybrid_gru', 'use_test_gru', 'use_standard_gru',
        'use_persistent_gru', 'use_sequential_triton_gru', 'use_selective_gru', 'use_projected_gru', 'h_recurrent', 'use_local_conv',
        'use_flash_gru', 'use_flash_ema_gru', 'use_cudnn_ema_gru', 'use_cudnn_multiscale_ema_gru',
        'use_cudnn_ssm_gru', 'use_cudnn_ssm_series_gru', 'per_layer_alpha', 'use_ema_gru', 'use_elman_silu', 'use_elman_leaky', 'use_elman_leaky_selective', 'use_leaky_elman', 'use_elman_leaky_silu', 'delta_init', 'use_haste_gru_silu', 'use_haste_gru_silu_fused', 'use_haste_lstm_silu', 'use_skip_elman_silu', 'use_elman_swish', 'use_elman_input_gate',
        'use_multihead_elman', 'multihead_elman_nheads', 'multihead_elman_headdim', 'multihead_elman_activation',
        'ema_alpha', 'use_gradient_checkpointing', 'z_bias_input', 'z_bias_hidden',
        'recurrence_chunk_size'
    }
    minlm_config = {k: v for k, v in model_config.items()
                    if k in minlm_valid_keys}
    return minLM(**minlm_config)

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
    parser.add_argument('--fresh_optimizer', action='store_true', help='skip loading optimizer state when resuming (resets Adam momentum/variance)')
    parser.add_argument('--save_every', type=int, default=2000, help='Target average interval (in steps) for one checkpoint to be saved across the entire population.')
    parser.add_argument('--batches_per_epoch', type=str, default="100", help='batches per epoch for dataloader length')
    parser.add_argument('--params', type=str, default="100m", help='target parameter count (e.g., 15m, 1g)')
    parser.add_argument('--dim', type=str, default=None, help='model hidden dimension (overrides params calculation)')
    parser.add_argument('--depth', type=int, default=None, help='number of layers (overrides params calculation)')
    parser.add_argument('--expansion_factor', type=float, default=1.5, help='state expansion factor for MinGRU inner dimension')
    parser.add_argument('--ff_mult', type=float, default=4.0, help='feedforward multiplier for MinGRU (ffn_dim = dim * ff_mult)')
    parser.add_argument('--conv_kernel_size', type=int, default=None, help='convolutional kernel size for preprocessing (None=disabled, typical: 4, 8, 16)')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for training (0.0=disabled)')
    parser.add_argument('--bf16', action='store_true', help='use bfloat16 mixed precision training')
    parser.add_argument('--no-autocast', action='store_true', dest='no_autocast', help='disable autocast when using bf16 (for CUDA kernels that handle BF16 natively)')
    parser.add_argument('--compile', action='store_true', help='use torch.compile for model optimization')
    parser.add_argument('--chunk_size', type=str, default="2k", help='sequence length of each chunk for BPTT')
    parser.add_argument('--batch_size', type=str, default="1", help='batch size per GPU (document streaming requires 1)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping threshold (L2 norm). Set to 0.0 to disable. WARNING: Clipping causes all-reduce every step, hurting throughput!')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    
    # Z-gate initialization parameters
    parser.add_argument('--z_bias_init', type=float, default=-2.0, 
                        help='Initial bias for z-gates (default: -2.0, more negative = more closed)')
    parser.add_argument('--z_bias_input', type=float, default=None,
                        help='Initial bias for z-gates on input projection (overrides z_bias_init)')
    parser.add_argument('--z_bias_hidden', type=float, default=None,
                        help='Initial bias for z-gates on hidden projection (overrides z_bias_init)')
    parser.add_argument('--recurrence_chunk_size', type=int, default=64,
                        help='Chunk size for GRU recurrence processing (trade-off: fewer kernel launches vs memory). Default=64 means 512/64=8 kernel launches instead of 512')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='number of recent checkpoints to keep')
    parser.add_argument('--keep_elite', type=int, default=10, help='number of elite models to preserve')
    parser.add_argument('--archive_rate', type=float, default=0.0, help='probability (0.0-1.0) of archiving checkpoints before deletion')
    parser.add_argument('--milestone_every', type=int, default=0, help='save permanent milestone checkpoint every N steps (0=disabled)')
    parser.add_argument('--no-schedulefree', dest='schedulefree', action='store_false', default=True)
    parser.add_argument('--sgd', action='store_true', help='use SGD with momentum instead of AdamW (simpler, no adaptive LR)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
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
    parser.add_argument('--gossip_fitness_window', type=int, default=1000,
                        help='Window size for fitness history tracking (kept for analysis but not used for fitness calculation).')
    parser.add_argument('--gossip-node-local-lock', dest='use_gossip_lock', action='store_true', default=False,
                        help='Enable a node-local lock to serialize gossip operations and prevent resource storms on multi-node systems.')
    parser.add_argument('--gossip_p_value_threshold', type=float, default=0.01,
                        help='P-value threshold for statistical significance in fitness comparison (default: 0.01).')
    parser.add_argument('--gossip_lock_timeout', type=float, default=2.0,
                        help='Timeout in seconds for gossip lock acquisition (default: 2.0)')
    parser.add_argument('--validation_interval', type=int, default=10000,
                        help='Steps between validation runs (default: 10000)')
    parser.add_argument('--validation_batches', type=int, default=8,
                        help='Number of batches to run for validation (default: 8)')
    
    # DDP (Distributed Data Parallel) support
    parser.add_argument('--ddp', action='store_true',
                        help='Enable DDP within nodes, gossip between nodes')
    parser.add_argument('--ddp-find-unused', action='store_true',
                        help='Enable find_unused_parameters in DDP (slower but safer)')
    parser.add_argument('--no-tbptt', action='store_true',
                        help='Disable TBPTT: reset hidden state each chunk (fair comparison with Mamba2)')

    # DataLoader configuration
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers for async data loading (default: 0, use 4-8 for pipelining)')
    
    # --- NEW: Filesystem-Augmented Evolution ---
    parser.add_argument('--filesystem-coordinator', action='store_true',
                        help='Enable filesystem-based coordination for rejuvenation and weighted checkpointing.')
    parser.add_argument('--rejuvenation-probability', type=float, default=0.001,
                        help='Base probability per step for a struggling model to load an elite checkpoint.')
    parser.add_argument('--rejuvenation-threshold', type=float, default=0.75,
                        help='Fitness percentile below which a model is considered "struggling" (e.g., 0.75 means bottom 25%).')
    parser.add_argument('--fitness-weighted-checkpointing', action='store_true',
                        help='Enable checkpointing probability based on fitness rank.')
    parser.add_argument('--elite-checkpoint-multiplier', type=float, default=4.0,
                        help='How much more likely top models are to save a checkpoint vs. the baseline.')
    parser.add_argument('--rejuvenation-tiebreaker-threshold', type=float, default=0.005,
                        help='If elite losses are within this fractional threshold, use step count as a tie-breaker (e.g., 0.01 for 1%).')
    
    # --- GRU Implementation Selection ---
    parser.add_argument('--fused_gru', '--hybrid_gru', action='store_true', dest='fused_gru',
                        help='Use Fused GRU (cuDNN kernel, 3× faster)')
    parser.add_argument('--use_test_gru', action='store_true',
                        help='Use simple test GRU implementation for debugging')
    parser.add_argument('--use_standard_gru', action='store_true',
                        help='Use PyTorch nn.GRU (cuDNN-optimized, gold standard nonlinear GRU)')
    parser.add_argument('--use_persistent_gru', action='store_true',
                        help='Use PersistentGRU (optimized persistent-T kernel, matches HybridGRU speed)')
    parser.add_argument('--use_sequential_triton_gru', action='store_true',
                        help='Use SequentialTritonGRU (non-persistent, no data race, works at 2K chunks)')
    parser.add_argument('--use_selective_gru', action='store_true',
                        help='Use SelectiveGRU (input-dependent Δ and selection, inspired by Mamba)')
    parser.add_argument('--use_projected_gru', action='store_true',
                        help='Use ProjectedGRU (reduced recurrent dim, 10%% faster + 67%% larger batch!)')
    parser.add_argument('--h_recurrent', type=int, default=None,
                        help='Recurrent dimension for ProjectedGRU (default: dim*0.625, e.g., 1280 for dim=2048)')
    parser.add_argument('--use_local_conv', action='store_true',
                        help='Use LocalConvGRU (local conv window, NOT recurrent!)')
    parser.add_argument('--use_flash_gru', action='store_true',
                        help='Use FlashRNN GRU (hardware-optimized, 50x speedup!)')
    parser.add_argument('--use_flash_ema_gru', action='store_true',
                        help='Use FlashRNN GRU + parallel EMA (60x faster + long-range memory!)')
    parser.add_argument('--use_cudnn_ema_gru', action='store_true',
                        help='Use cuDNN GRU + parallel EMA (DDP-compatible + long-range memory!)')
    parser.add_argument('--use_cudnn_multiscale_ema_gru', action='store_true',
                        help='Use cuDNN GRU + Multi-Scale EMA (3 timescales: 0.1/0.01/0.001)')
    parser.add_argument('--use_cudnn_ssm_gru', action='store_true',
                        help='Use cuDNN GRU + Selective Diagonal SSM (learned decay, input-dependent like Mamba)')
    parser.add_argument('--use_cudnn_ssm_series_gru', action='store_true',
                        help='Use cuDNN GRU -> SSM in series (GRU extracts features, SSM tracks state)')
    parser.add_argument('--per_layer_alpha', action='store_true',
                        help='Initialize each layer with different EMA alpha (fast→slow with depth)')
    parser.add_argument('--use_ema_gru', action='store_true',
                        help='Use EMA GRU (GRU + EMA for long-range memory)')
    parser.add_argument('--use_elman_silu', action='store_true',
                        help='Use ElmanSilu (haste CUDA kernels, 3x faster than cuDNN GRU!)')
    parser.add_argument('--use_elman_leaky', action='store_true',
                        help='Use ElmanLeaky (true discretized dynamics, input-dependent delta!)')
    parser.add_argument('--use_elman_leaky_selective', action='store_true',
                        help='Use ElmanLeakySelective (Mamba2-style discretization + h+x output gate!)')
    parser.add_argument('--use_leaky_elman', action='store_true',
                        help='Use LeakyElman (leaky integration + INPUT-ONLY output gate)')
    parser.add_argument('--use_elman_leaky_silu', action='store_true',
                        help='Use ElmanLeakySilu (silu + leaky integration, like ElmanLeaky but with silu!)')
    parser.add_argument('--use_haste_gru_silu', action='store_true',
                        help='Use HasteGRUSilu (haste GRU + silu gate, proper skip connection like cuDNN!)')
    parser.add_argument('--use_haste_gru_silu_fused', action='store_true',
                        help='Use HasteGRUSiluFused (fused GRU+silu CUDA kernel, BF16 native!)')
    parser.add_argument('--use_haste_lstm_silu', action='store_true',
                        help='Use HasteLSTMSilu (fused LSTM+silu CUDA kernel, BF16 native!)')
    parser.add_argument('--use_skip_elman_silu', action='store_true',
                        help='Use SkipElmanSilu (SkipElman + silu gate, simpler than GRU!)')
    parser.add_argument('--use_elman_swish', action='store_true',
                        help='Use ElmanSwish (silu inside + silu gate, SwiGLU-style like Mamba2!)')
    parser.add_argument('--use_elman_input_gate', action='store_true',
                        help='Use ElmanInputGate (input-only gating like Mamba2, better gradient flow!)')
    parser.add_argument('--use_multihead_elman', action='store_true',
                        help='Use MultiHeadElman (32 heads × 64×64 R matrices, 2048x more expressive than Mamba2!)')
    parser.add_argument('--multihead_elman_nheads', type=int, default=32,
                        help='Number of heads for MultiHeadElman (default 32)')
    parser.add_argument('--multihead_elman_headdim', type=int, default=64,
                        help='Dimension per head for MultiHeadElman (default 64)')
    parser.add_argument('--multihead_elman_activation', type=str, default='softsign',
                        choices=['softsign', 'tanh_residual', 'tanh'],
                        help='Activation for MultiHeadElman: softsign (gradient-friendly), tanh_residual, or tanh')
    parser.add_argument('--ema_alpha', type=float, default=0.01,
                        help='EMA decay rate (small = longer memory, 0.01 ~ 70 token half-life)')
    parser.add_argument('--use_deep_normal_gru', action='store_true',
                        help='Use DeepNormalGRULM (RECOMMENDED: normal-space + residuals + LayerNorm + identity init for deep 20-32 layer networks)')
    parser.add_argument('--use_logspace_gru', action='store_true',
                        help='Use DeepLogSpaceGRULM (EXPERIMENTAL: log-space hidden states, unstable training)')
    parser.add_argument('--use_mamba', action='store_true',
                        help='Use Mamba SSM (gold standard for RNNs, requires mamba-ssm package)')
    parser.add_argument('--use_mamba2', action='store_true',
                        help='Use Mamba2 SSM with SSD (faster parallel training, requires mamba-ssm package)')
    parser.add_argument('--use_hybrid_mamba2_gru', action='store_true',
                        help='Use Hybrid Mamba2+cuDNN GRU (parallel paths with learned mixing, no TBPTT)')
    parser.add_argument('--use_mamba2_ffn3', action='store_true',
                        help='Use Mamba2 + 3-layer FFN (arXiv:2505.06633: d→4d→4d→d with GELU, no TBPTT)')
    parser.add_argument('--use_cudnn_conv_gru', action='store_true',
                        help='Use cuDNN GRU + Causal Conv1d (Mamba2-style 4-wide local context before GRU)')
    parser.add_argument('--use_cudnn_mult_gru', action='store_true',
                        help='Use cuDNN GRU + Multiplicative Gating (adds h*f(x,h) nonlinearity after GRU)')
    parser.add_argument('--use_haste_mult_gru', action='store_true',
                        help='Use Haste GRU + SiLU Multiplicative Gating (fused CUDA kernel, BF16 native, EXACT numerical match to CuDNN+silu)')
    parser.add_argument('--use_discretized_elman', action='store_true',
                        help='Use Discretized Elman with explicit delta (leaky integrator with input-dependent step size)')
    parser.add_argument('--delta_mode', type=str, default='fixed',
                        choices=['fixed', 'learned', 'input', 'input_state'],
                        help='Delta computation mode for discretized Elman')
    parser.add_argument('--delta_init', type=float, default=-2.0,
                        help='Delta initialization for ElmanLeaky/DiscretizedElman (sigmoid(-2)≈0.12 for slow dynamics)')
    parser.add_argument('--use_mingru_mult', action='store_true',
                        help='Use minGRU + SiLU output gate (parallel scan + selectivity, like fast Mamba+GRU)')
    parser.add_argument('--use_cudnn_lstm_mult', action='store_true',
                        help='Use cuDNN LSTM + SiLU output gate (more stepwise nonlinearity than GRU)')
    parser.add_argument('--use_elman_selective', action='store_true',
                        help='Use Elman MLP + SiLU selectivity (simplest recurrence: MLP state mixing + output gating)')
    parser.add_argument('--use_cudnn_bilinear_gru', action='store_true',
                        help='Use cuDNN GRU + Bilinear interactions (true second-order h×x terms)')
    parser.add_argument('--use_cudnn_plain_gru', action='store_true',
                        help='Use plain cuDNN GRU baseline (no modifications, pure GRU + residual)')
    parser.add_argument('--use_cudnn_ffn3_gru', action='store_true',
                        help='Use cuDNN GRU + 3-layer FFN selectivity gate (deep selectivity on concat[x,h])')
    parser.add_argument('--ffn_expand', type=float, default=2.0,
                        help='For FFN3 GRU: hidden dim multiplier for FFN gate (default 2.0)')
    parser.add_argument('--use_ema_input_gate', action='store_true',
                        help='Use EMA + input-dependent gate (minimal selectivity test)')
    parser.add_argument('--input_only_gate', action='store_true',
                        help='For Mult GRU: gate depends only on input x, not hidden h (ablation test)')
    parser.add_argument('--use_glu_gate', action='store_true',
                        help='For Mult GRU: use GLU-style gate (split h, gate with itself, no x dependence)')
    parser.add_argument('--use_swiglu', action='store_true',
                        help='For Mult GRU: use SwiGLU-style output (y = value * SiLU(gate + x), same params)')
    parser.add_argument('--gate_activation', type=str, default='sigmoid', choices=['sigmoid', 'silu'],
                        help='For Mult GRU: gate activation function (sigmoid default, silu like Mamba2)')
    parser.add_argument('--use_checkpointing', action='store_true',
                        help='Enable gradient checkpointing for CuDNN Mult GRU (trades compute for memory)')
    parser.add_argument('--inner_chunk_size', type=int, default=128,
                        help='Chunk size for gradient checkpointing in Mult GRU (default 128)')
    parser.add_argument('--mamba_d_state', type=int, default=16,
                        help='Mamba SSM state dimension (default 16 for Mamba, 64 for Mamba2)')
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba expansion factor (default 2)')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to reduce memory (trades compute for memory)')

    # --- Zero-Order Optimization (CD-RGE) ---
    zo_group = parser.add_argument_group('Zero-Order Optimization')
    zo_group.add_argument(
        '--zero_order',
        action='store_true',
        help='Enable zero-order optimization (CD-RGE) for memory-efficient training'
    )
    zo_group.add_argument(
        '--zo_method',
        type=str,
        default='cd_rge',
        choices=['cd_rge', 'layerwise', 'mezo'],
        help='Zero-order method: cd_rge (sequential), layerwise (parallel, 7× faster), mezo (memory-efficient, 2 forward passes)'
    )
    zo_group.add_argument(
        '--zo_n_perturbations',
        type=int,
        default=96,
        help='Number of probe vectors for gradient estimation (default: 96, range: 48-512)'
    )
    zo_group.add_argument(
        '--zo_perturbation_chunk_size',
        type=int,
        default=8,
        help='Number of perturbations to materialize at once for layerwise method (default: 8, reduces OOM risk)'
    )
    zo_group.add_argument(
        '--zo_epsilon',
        type=float,
        default=None,
        help='Perturbation size for CD-RGE (default: equal to learning rate for stability)'
    )
    zo_group.add_argument(
        '--zo_num_perturbations_mezo',
        type=int,
        default=4,
        help='Number of perturbations per step for MeZO (K). Default: 4. Lower K = faster but higher variance. Each perturbation = 2 forward passes.'
    )
    zo_group.add_argument(
        '--zo_probe_distribution',
        type=str,
        default='rademacher',
        choices=['rademacher', 'gaussian'],
        help='Probe distribution (rademacher recommended for high dimensions)'
    )
    zo_group.add_argument(
        '--zo_memory_chunk',
        type=int,
        default=512,
        help='Chunk size for memory-efficient forward passes (default: 512 tokens)'
    )

    # --- Tokenization Options ---
    tokenizer_group = parser.add_argument_group('Tokenization')
    tokenizer_group.add_argument(
        '--tokenizer',
        type=str,
        default='byte',
        choices=['byte', 'tiktoken', 'sentencepiece', 'huggingface'],
        help='Tokenizer type (default: byte for backwards compatibility)'
    )
    tokenizer_group.add_argument(
        '--tiktoken_encoding',
        type=str,
        default='cl100k_base',
        choices=['cl100k_base', 'p50k_base', 'r50k_base', 'o200k_base'],
        help='TikToken encoding name (default: cl100k_base, GPT-3.5/4 tokenizer with 100K vocab)'
    )
    tokenizer_group.add_argument(
        '--sentencepiece_model',
        type=str,
        default=None,
        help='Path to SentencePiece .model file (required if --tokenizer=sentencepiece)'
    )
    tokenizer_group.add_argument(
        '--huggingface_tokenizer',
        type=str,
        default=None,
        help='HuggingFace tokenizer name (e.g., "meta-llama/Llama-2-7b-hf", required if --tokenizer=huggingface)'
    )

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument('--cuda', action='store_true')
    backend_group.add_argument('--rocm', action='store_true')
    args = parser.parse_args()
    return args

@torch.no_grad()
def measure_z_stats(model, batch_x, hidden_state_snapshot=None, layer_index=0, norm_first=True):
    """
    Returns a dict with z gate stats for a given layer.
    - batch_x: LongTensor tokens [B, T] from your current stream (small slice OK)
    - hidden_state_snapshot: optional List[Tensor] like your 'hidden_state'; if None, uses zeros(h)
    - layer_index: which layer's GRU to probe (0 = first)
    - norm_first: apply that layer's RMSNorm before projection (matches forward path)
    """
    # 1) grab layer modules
    layer = model.layers[layer_index]
    norm = layer[1]
    gru  = layer[2]  # HybridFusedGRU or minGRU

    device = next(model.parameters()).device

    # 2) embed + optional norm, then one step (use first token for speed / clarity)
    x = model.token_emb(batch_x.to(device))[:, :1]  # [B, 1, D]
    if norm_first:
        x = norm(x)

    # 3) choose h_{t-1}
    if hasattr(gru, 'dim_inner'):
        H = gru.dim_inner
    else:
        H = gru.to_hidden_and_gate.out_features // 2  # minGRU path

    if hidden_state_snapshot is not None and len(hidden_state_snapshot) > layer_index:
        h_prev = hidden_state_snapshot[layer_index]
        if h_prev.ndim == 3:  # sometimes [B,1,H]
            h_prev = h_prev.squeeze(1)
    else:
        h_prev = torch.zeros(x.size(0), H, device=device, dtype=x.dtype)

    # 4) compute gates via the model's own projections
    # NOTE: for HybridFusedGRU: input_projection: D -> 3H, hidden_projection: H -> 3H
    if hasattr(gru, 'input_projection') and hasattr(gru, 'hidden_projection'):
        inp = gru.input_projection(x.squeeze(1))      # [B, 3H]
        hid = gru.hidden_projection(h_prev)           # [B, 3H]
        # split: [r | z | n]
        _, i_z, _ = inp.chunk(3, dim=-1)
        _, h_z, _ = hid.chunk(3, dim=-1)
        z = torch.sigmoid(i_z + h_z)
    else:
        # minGRU: single Linear to [hidden, gate], gate is the 'z'
        hidden, gate = gru.to_hidden_and_gate(x.squeeze(1)).chunk(2, dim=-1)
        z = torch.sigmoid(gate)  # effective openness
    # 5) summarize
    zf = z.float()
    q = torch.quantile(zf, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=zf.device))
    return {
        "mean": float(zf.mean().item()),
        "std":  float(zf.std(unbiased=False).item()),
        "p01":  float(q[0].item()),
        "p10":  float(q[1].item()),
        "p50":  float(q[2].item()),
        "p90":  float(q[3].item()),
        "p99":  float(q[4].item()),
    }

def log_z_stats_to_tsv(model, batch_x, hidden_state, step, tsv_file, num_layers_to_log=None):
    """
    Log z-gate statistics for multiple layers to a TSV file.
    - model: the model to analyze
    - batch_x: current batch tokens [B, T] 
    - hidden_state: current hidden state snapshot
    - step: current training step
    - tsv_file: path to TSV file
    - num_layers_to_log: how many layers to log (None = all)
    """
    import time
    
    # Extract model from DDP wrapper if needed
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Determine how many layers to log
    total_layers = len(actual_model.layers)
    layers_to_log = min(num_layers_to_log or total_layers, total_layers)
    
    # Use a small probe from current batch
    probe_tokens = batch_x[:, :32].detach()
    
    # Write header if file doesn't exist
    write_header = not os.path.exists(tsv_file)
    
    with open(tsv_file, 'a') as f:
        if write_header:
            # Header: step, timestamp, layer, type, then all stats
            f.write("step\ttimestamp\tlayer\ttype\tmean\tstd\tp01\tp10\tp50\tp90\tp99\n")
        
        timestamp = time.time()
        
        for layer_idx in range(layers_to_log):
            # Runtime z (with current hidden state)
            stats_runtime = measure_z_stats(actual_model, probe_tokens, 
                                           hidden_state_snapshot=hidden_state, 
                                           layer_index=layer_idx)
            
            # Input-only z (with zero hidden state)
            stats_input = measure_z_stats(actual_model, probe_tokens, 
                                         hidden_state_snapshot=None, 
                                         layer_index=layer_idx)
            
            # Write runtime stats
            f.write(f"{step}\t{timestamp:.3f}\t{layer_idx}\truntime\t"
                   f"{stats_runtime['mean']:.6f}\t{stats_runtime['std']:.6f}\t"
                   f"{stats_runtime['p01']:.6f}\t{stats_runtime['p10']:.6f}\t"
                   f"{stats_runtime['p50']:.6f}\t{stats_runtime['p90']:.6f}\t"
                   f"{stats_runtime['p99']:.6f}\n")
            
            # Write input-only stats
            f.write(f"{step}\t{timestamp:.3f}\t{layer_idx}\tinput\t"
                   f"{stats_input['mean']:.6f}\t{stats_input['std']:.6f}\t"
                   f"{stats_input['p01']:.6f}\t{stats_input['p10']:.6f}\t"
                   f"{stats_input['p50']:.6f}\t{stats_input['p90']:.6f}\t"
                   f"{stats_input['p99']:.6f}\n")
        
        f.flush()

def main():
    args = get_args()

    # --- 1. DISTRIBUTED INITIALIZATION ---
    configure_backend(args)
    
    # Make rank discovery compatible with both deepspeed launcher and srun
    global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NPROCS', '1')))

    
    # --- START OF DEFINITIVE FIX ---
    # When using srun with --gpus-per-task, each process sees only one GPU, indexed at 0.
    # We must set the device to 0 for all ranks in that case.
    # The deepspeed launcher exposes all GPUs, so we use local_rank there.
    # This logic handles both cases robustly.
    if torch.cuda.is_available():
        visible_devices = torch.cuda.device_count()
        if visible_devices == 1:
            # srun --gpus-per-task=1 case: The single visible device is always at index 0.
            device_id = 0
        else:
            # Deepspeed or other launchers: Use local_rank to select from multiple visible devices.
            device_id = local_rank
        
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
    else:
        # Fallback for CPU-only testing
        device = torch.device('cpu')
    # --- END OF DEFINITIVE FIX ---

    random.seed(SEED + global_rank)

    # --- FIX: ROBUST DIRECTORY CREATION and Debugging ---
    # Rank 0 creates the output directory, all others wait for it to be ready.
    if global_rank == 0:
        print(f"\nRe-seeded Python's random module per-rank to ensure stochastic mixing.\n")

    # --- 2. CONFIGURATION AND SETUP ---
    train_steps = int(parse_size_with_suffix(args.train_steps))
    chunk_size = int(parse_size_with_suffix(args.chunk_size))
    batch_size = int(parse_size_with_suffix(args.batch_size))
    batches_per_epoch = int(parse_size_with_suffix(args.batches_per_epoch))
    
    resuming = args.resume is not None
    resume_step = 0
    # Use args.output directly, which is now guaranteed to exist.
    checkpoint_dir = args.output or f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Rank 0 creates the checkpoint directory and subdirectories, all others wait for it to be ready.
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'gossip'), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'metrics'), exist_ok=True)

    if world_size > 1:
        simple_barrier('setup')
    
    # Set environment variable for memory logging
    os.environ['GRUBOROS_OUTPUT_DIR'] = checkpoint_dir

    # DISABLED: No background thread, using synchronous checkpoint management
    # checkpoint_manager = CheckpointManager(
    #     checkpoint_dir=checkpoint_dir, keep_last_n=args.keep_checkpoints,
    #     keep_elite_n=args.keep_elite, global_rank=global_rank, archive_rate=args.archive_rate,
    #     milestone_every=args.milestone_every
    # )
    # checkpoint_manager.start()

    model_config, checkpoint = None, None
    if resuming:
        path = args.resume if os.path.isfile(args.resume) else os.path.join(args.resume, "latest.pt")
        if os.path.exists(path):
            if global_rank == 0: print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location='cpu')
            model_config, resume_step = checkpoint.get('model_config'), checkpoint.get('step', 0)
            
            # Backwards compatibility for old checkpoints
            if model_config is not None:
                if 'enable_conv' in model_config:
                    # Old style - convert to new
                    if model_config.pop('enable_conv'):
                        # Old conv was enabled with default kernel size 3
                        if 'conv_kernel_size' not in model_config:
                            model_config['conv_kernel_size'] = 3
                    else:
                        model_config['conv_kernel_size'] = None
                elif 'conv_kernel_size' not in model_config:
                    # Very old checkpoint without any conv config
                    model_config['conv_kernel_size'] = None

    if model_config is None:
        params_value = parse_size_with_suffix(args.params)
        
        if args.dim and args.depth:
            # User specified both, use them directly
            dim = int(parse_size_with_suffix(args.dim))
            depth = args.depth
        elif args.dim and not args.depth:
            # User specified dim, solve for depth
            dim = int(parse_size_with_suffix(args.dim))
            depth = solve_for_depth(params_value, dim, expansion=args.expansion_factor, ff_mult=args.ff_mult, use_hybrid_gru=args.fused_gru)
        elif not args.dim and args.depth:
            # User specified depth, solve for dim
            depth = args.depth
            dim = solve_for_dimension(params_value, depth, expansion=args.expansion_factor, ff_mult=args.ff_mult, use_hybrid_gru=args.fused_gru)
        else:
            # Default behavior: guess a dim and solve for depth, then refine dim
            base_dim = 512 if params_value < 1e9 else 1024
            # Heuristic scaling for dimension based on Chinchilla laws (very approximate)
            dim_guess = round_to_multiple(base_dim * (params_value / (100e6 if params_value < 1e9 else 1e9))**0.25)
            depth = solve_for_depth(params_value, dim_guess, expansion=args.expansion_factor, ff_mult=args.ff_mult, use_hybrid_gru=args.fused_gru)
            dim = solve_for_dimension(params_value, depth, expansion=args.expansion_factor, ff_mult=args.ff_mult, use_hybrid_gru=args.fused_gru)
            
        model_config = {
            "num_tokens": 256,  # Will be updated by tokenizer
            "dim": dim,
            "depth": depth,
            "ff_mult": args.ff_mult,
            "expansion": args.expansion_factor,
            "conv_kernel_size": args.conv_kernel_size,
            "dropout": args.dropout,
            "use_fused_gru": args.fused_gru,
            "use_test_gru": args.use_test_gru,
            "use_standard_gru": args.use_standard_gru,
            "use_persistent_gru": args.use_persistent_gru,
            "use_sequential_triton_gru": args.use_sequential_triton_gru,
            "use_selective_gru": args.use_selective_gru,
            "use_projected_gru": args.use_projected_gru,
            "h_recurrent": args.h_recurrent,
            "use_local_conv": args.use_local_conv,
            "use_flash_gru": args.use_flash_gru,
            "use_flash_ema_gru": args.use_flash_ema_gru,
            "use_cudnn_ema_gru": args.use_cudnn_ema_gru,
            "use_cudnn_multiscale_ema_gru": args.use_cudnn_multiscale_ema_gru,
            "use_cudnn_ssm_gru": args.use_cudnn_ssm_gru,
            "use_cudnn_ssm_series_gru": args.use_cudnn_ssm_series_gru,
            "per_layer_alpha": args.per_layer_alpha,
            "use_ema_gru": args.use_ema_gru,
            "use_elman_silu": args.use_elman_silu,
            "use_elman_leaky": args.use_elman_leaky,
            "use_elman_leaky_selective": args.use_elman_leaky_selective,
            "use_leaky_elman": args.use_leaky_elman,
            "use_elman_leaky_silu": args.use_elman_leaky_silu,
            "delta_init": args.delta_init,
            "use_haste_gru_silu": args.use_haste_gru_silu,
            "use_haste_gru_silu_fused": args.use_haste_gru_silu_fused,
            "use_haste_lstm_silu": args.use_haste_lstm_silu,
            "use_skip_elman_silu": args.use_skip_elman_silu,
            "use_elman_swish": args.use_elman_swish,
            "use_elman_input_gate": args.use_elman_input_gate,
            "use_multihead_elman": args.use_multihead_elman,
            "multihead_elman_nheads": args.multihead_elman_nheads,
            "multihead_elman_headdim": args.multihead_elman_headdim,
            "multihead_elman_activation": args.multihead_elman_activation,
            "ema_alpha": args.ema_alpha,
            "use_deep_normal_gru": args.use_deep_normal_gru,
            "use_logspace_gru": args.use_logspace_gru,
            "use_mamba": args.use_mamba,
            "use_mamba2": args.use_mamba2,
            "use_hybrid_mamba2_gru": args.use_hybrid_mamba2_gru,
            "use_mamba2_ffn3": args.use_mamba2_ffn3,
            "use_cudnn_conv_gru": args.use_cudnn_conv_gru,
            "use_cudnn_mult_gru": args.use_cudnn_mult_gru,
            "use_haste_mult_gru": args.use_haste_mult_gru,
            "use_discretized_elman": args.use_discretized_elman,
            "delta_mode": args.delta_mode,
            "delta_init": args.delta_init,
            "use_mingru_mult": args.use_mingru_mult,
            "use_cudnn_lstm_mult": args.use_cudnn_lstm_mult,
            "use_elman_selective": args.use_elman_selective,
            "use_cudnn_bilinear_gru": args.use_cudnn_bilinear_gru,
            "use_cudnn_plain_gru": args.use_cudnn_plain_gru,
            "use_cudnn_ffn3_gru": args.use_cudnn_ffn3_gru,
            "ffn_expand": args.ffn_expand,
            "use_ema_input_gate": args.use_ema_input_gate,
            "input_only_gate": args.input_only_gate,
            "use_glu_gate": args.use_glu_gate,
            "use_swiglu": args.use_swiglu,
            "gate_activation": args.gate_activation,
            "use_checkpointing": args.use_checkpointing,
            "inner_chunk_size": args.inner_chunk_size,
            "mamba_d_state": args.mamba_d_state,
            "mamba_expand": args.mamba_expand,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "z_bias_input": args.z_bias_input if args.z_bias_input is not None else args.z_bias_init,
            "z_bias_hidden": args.z_bias_hidden if args.z_bias_hidden is not None else args.z_bias_init,
            "recurrence_chunk_size": args.recurrence_chunk_size
        }

    # Initialize tokenizer BEFORE creating model (vocab size needed for model config)
    from mingru.tokenizers import get_tokenizer

    if args.tokenizer == 'byte':
        tokenizer = get_tokenizer('byte')
    elif args.tokenizer == 'tiktoken':
        tokenizer = get_tokenizer('tiktoken', encoding_name=args.tiktoken_encoding)
    elif args.tokenizer == 'sentencepiece':
        if not args.sentencepiece_model:
            raise ValueError("--sentencepiece_model required when --tokenizer=sentencepiece")
        tokenizer = get_tokenizer('sentencepiece', model_path=args.sentencepiece_model)
    elif args.tokenizer == 'huggingface':
        if not args.huggingface_tokenizer:
            raise ValueError("--huggingface_tokenizer required when --tokenizer=huggingface")
        tokenizer = get_tokenizer('huggingface', tokenizer_name=args.huggingface_tokenizer)

    # Update model config with actual vocab size
    model_config["num_tokens"] = tokenizer.vocab_size

    if global_rank == 0:
        print(f"\n=== Tokenization ===")
        print(f"Tokenizer: {tokenizer}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print("====================\n")

    if global_rank == 0:
        print(f"Model size: {get_parameter_count_str(model_config)} parameters")
        print(f"Configuration: {model_config}")
        # The directory is now guaranteed to exist before this is called.
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        ### DOCUMENT STREAMING TRAINING DYNAMICS ###
        print("\n--- Document Streaming Training ---")
        print(f"Document delimiter: 0x1e (each GPU reads different documents)")
        print(f"Chunk size: {chunk_size} tokens")
        print(f"Dynamic optimization: at document boundaries OR every {args.grad_accum} chunks (whichever comes first)")
        print(f"Hidden state: resets at document boundaries for proper context")
        print(f"Per-GPU token tracking: each GPU processes different portions of the dataset")
        print("-------------------------------------\n")


    model = get_model(model_config).to(device)

    # Convert model to bf16 if using mixed precision
    # This is required for FlashRNN which JIT compiles based on weight dtype
    if args.bf16:
        model = model.bfloat16()
        if global_rank == 0:
            print("Model converted to bfloat16 for FlashRNN compatibility")

    # Print ACTUAL parameter count (not estimated!)
    if global_rank == 0:
        actual_params = sum(p.numel() for p in model.parameters())
        print(f"ACTUAL model parameters: {actual_params/1e9:.2f}B ({actual_params/1e6:.1f}M)")

    # MEMORY CHECKPOINT 1: After model creation
    if global_rank == 0 and device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"[MEMORY] After model creation: {mem_allocated:.2f} GB")

    # Compile the model for better performance
    if args.compile:
        if global_rank == 0:
            print("\n=== Compiling model with torch.compile ===")
            print("This may take several minutes on first run, but is cached for subsequent runs.")
        # Use reduce-overhead mode for better performance, fullgraph=False for compatibility
        # Cache is automatically enabled via TORCHINDUCTOR_CACHE_DIR (set in shell)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # Initialize optimizer based on training mode
    if args.zero_order:
        # Set epsilon = lr if not explicitly specified (paper recommendation)
        epsilon = args.zo_epsilon if args.zo_epsilon is not None else args.lr

        if args.zo_method == 'mezo':
            from mezo_batched_optimizer import MeZOBatchedOptimizer

            optimizer = MeZOBatchedOptimizer(
                model=model,
                learning_rate=args.lr,
                epsilon=epsilon,
                batch_size=batch_size,  # Perturbations per GPU processed in PARALLEL!
                base_seed=42,
                rank=global_rank,
                world_size=world_size,
                momentum=args.sf_beta,  # Simple SGD momentum
                grad_accum=args.grad_accum,  # Gradient accumulation for stability
            )

            # NOTE: We do NOT use DDP for MeZO! DDP allocates gradient buffers we don't need.
            # MeZO uses dist.all_gather() for gradient coefficient cooperation (in batched optimizer).
            # This will be handled by skipping DDP wrapping below.

            if global_rank == 0:
                total_perturbations = batch_size * world_size
                print("\n=== Zero-Order Optimization (MeZO BATCHED) ===")
                print(f"Method: BATCHED parallel perturbations with seed-based generation")
                print(f"Learning rate: {args.lr}")
                print(f"Epsilon: {epsilon}")
                print(f"Perturbations per GPU: {batch_size} (processed in PARALLEL via batch dimension!)")
                print(f"Total perturbations: {total_perturbations} ({batch_size} × {world_size} GPUs)")
                print(f"Forward passes per GPU: {2 * batch_size} (vs {2 * args.zo_num_perturbations_mezo} serial in old version)")
                print(f"Expected speedup: ~{args.zo_num_perturbations_mezo // batch_size}× vs serial K={args.zo_num_perturbations_mezo}")
                print(f"Gradient accumulation steps: {args.grad_accum}")
                print(f"Memory: Same as inference (no gradients, no backward)")
                print(f"GPU cooperation: dist.all_gather() for gradient coefficients")
                print(f"requires_grad=False (prevents DDP gradient buffer allocation!)")
                print("================================================\n")

                # MEMORY CHECKPOINT 2: After optimizer creation
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[MEMORY] After optimizer creation: {mem_allocated:.2f} GB")

        elif args.zo_method == 'layerwise':
            from zero_order_layerwise import LayerwiseZeroOrderOptimizer

            optimizer = LayerwiseZeroOrderOptimizer(
                model=model,
                learning_rate=args.lr,
                epsilon=epsilon,
                n_perturbations=args.zo_n_perturbations,
                perturbation_chunk_size=args.zo_perturbation_chunk_size,
                base_seed=42,
                rank=global_rank
            )

            if global_rank == 0:
                print("\n=== Zero-Order Optimization (Layer-wise Parallel) ===")
                print(f"Method: Layer-wise materialized perturbations (7× faster)")
                print(f"Learning rate: {args.lr}")
                print(f"Epsilon: {epsilon}")
                print(f"Perturbations: {args.zo_n_perturbations}")
        else:
            from zero_order_optimizer import CD_RGE_Optimizer, ZeroOrderLossWrapper

            optimizer = CD_RGE_Optimizer(
                model=model,
                learning_rate=args.lr,
                epsilon=epsilon,
                n_perturbations=args.zo_n_perturbations,
                world_size=world_size,
                rank=global_rank,
                chunk_size=args.zo_memory_chunk,
                grad_accum=args.grad_accum
            )

            if global_rank == 0:
                print("\n=== Zero-Order Optimization (CD-RGE) ===")
                print(f"Method: Sequential virtual perturbations")
                print(f"Learning rate: {args.lr}")
                print(f"Epsilon: {epsilon}")
                print(f"Perturbations: {args.zo_n_perturbations}")
            print(f"Forward passes per step: {2 * args.zo_n_perturbations}")
            print(f"Probe distribution: {args.zo_probe_distribution}")
            print("=========================================\n")
    else:
        if args.sgd:
            # Simple SGD with momentum - no adaptive learning rate, what you set is what you get
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            if global_rank == 0: print(f"Using SGD optimizer (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
        elif args.schedulefree:
            optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
            if global_rank == 0: print(f"Using Schedule-Free AdamW optimizer")
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
            if global_rank == 0: print(f"Using AdamW optimizer")
    
    # FlashRNN workaround: Scale loss to avoid bias gradient kernel bug in backward pass
    # The bug only triggers with certain gradient magnitude ranges
    # We scale the loss, then manually divide gradients before optimizer step
    use_flashrnn = getattr(args, 'use_flash_gru', False) or getattr(args, 'use_flash_ema_gru', False)
    flashrnn_loss_scale = 65536.0 if use_flashrnn and args.bf16 else 1.0
    scaler = None  # Don't use GradScaler (it doesn't support bf16 unscale)
    
    if resuming and checkpoint:
        # Handle DDP module. prefix mismatch
        # Checkpoint might have been saved with DDP (module. prefix), but we load before wrapping
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            # Strip module. prefix
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if args.fresh_optimizer:
            # Skip loading optimizer state - start with fresh Adam momentum/variance
            if global_rank == 0: print(f"Resumed model from step {resume_step} with FRESH optimizer (momentum/variance reset)")
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if global_rank == 0: print(f"Resumed model and optimizer from step {resume_step}")

    if args.schedulefree and not args.zero_order and not args.sgd: optimizer.train()

    # Pre-compile FlashRNN kernels BEFORE DDP init (avoids barrier timeout)
    # FlashRNN JIT compilation can take 10-15 minutes for large dimensions,
    # which exceeds NCCL timeout. We use file-based synchronization instead.
    use_flashrnn = getattr(args, 'use_flash_gru', False) or getattr(args, 'use_flash_ema_gru', False)
    if use_flashrnn and args.ddp:
        from mingru.flash_gru import precompile_flashrnn_kernels
        # dim/batch_size may be nested lists if specified as positional args, handle robustly
        dim_val = args.dim
        while isinstance(dim_val, (list, tuple)):
            dim_val = dim_val[0]
        dim_val = int(dim_val)
        batch_val = args.batch_size
        while isinstance(batch_val, (list, tuple)):
            batch_val = batch_val[0]
        batch_val = int(batch_val)
        expansion = getattr(args, 'expansion_factor', 1.0)
        head_dim = int(dim_val * expansion)
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        print(f"[DEBUG] FlashRNN precompile: dim_val={dim_val} type={type(dim_val)}, batch_val={batch_val} type={type(batch_val)}, head_dim={head_dim} type={type(head_dim)}", flush=True)
        precompile_flashrnn_kernels(dim_inner=head_dim, batch_size=batch_val, device=device, dtype=dtype)

    # Setup DDP if enabled
    if args.ddp:
        # Setup DDP groups (needed for process group initialization)
        ddp_group, ddp_rank, ddp_world_size, is_ddp_primary, node_id = setup_ddp_groups(
            global_rank, local_rank, world_size
        )

        if not args.zero_order:
            # Standard training: Wrap model in DDP
            if global_rank == 0:
                print(f"\n=== DDP Configuration ===")
                print(f"DDP enabled: {ddp_world_size} ranks per node")
                print(f"Node {node_id}: ranks {global_rank - ddp_rank} to {global_rank - ddp_rank + ddp_world_size - 1}")
                print(f"Gossip participants: rank 0 from each node")
                print("=========================\n")

            # Check if using FlashRNN which requires simpler DDP settings
            use_flashrnn = getattr(args, 'use_flash_gru', False) or getattr(args, 'use_flash_ema_gru', False)

            model = DDP(
                model,
                device_ids=[device_id] if device.type == 'cuda' else None,
                process_group=ddp_group,
                find_unused_parameters=args.ddp_find_unused,
                # Disable optimizations for FlashRNN compatibility - its JIT kernels conflict with these
                gradient_as_bucket_view=not use_flashrnn,
                static_graph=args.ddp_find_unused and not use_flashrnn
            )

            # For DDP, we need to access the underlying module for gossip
            base_model = model.module
        else:
            # Zero-order: NO DDP WRAPPING! Saves MASSIVE memory by avoiding gradient buffers.
            # MeZO only needs dist.all_reduce() for scalar values (process group already initialized above).
            base_model = model

            if global_rank == 0:
                print("\n[MeZO] SKIPPING DDP MODEL WRAPPING (saves ~15+ GB gradient buffers!)")
                print(f"[MeZO] Using {world_size} independent processes with scalar synchronization")
                print()

    else:
        ddp_group = None
        ddp_rank = 0
        ddp_world_size = 1
        is_ddp_primary = True
        node_id = global_rank
        base_model = model

    # Create batched document streaming dataset
    # For DDP, ensure each rank gets different data
    # IMPORTANT: Incorporate resume_step to avoid data repetition when resuming!
    if args.ddp:
        # Each rank should see different data, offset by resume_step to avoid repetition
        dataset_seed = SEED + global_rank * 1000 + resume_step
    else:
        dataset_seed = SEED + global_rank * 1000 + resume_step

    # DEBUG: Verify tokenizer before passing
    print(f"[RANK {global_rank}] Creating dataset with tokenizer: {tokenizer}")
    print(f"[RANK {global_rank}] Tokenizer class: {tokenizer.__class__.__name__}")

    train_dataset = DocumentStreamWrapper(
        args.data,
        chunk_size=chunk_size,
        batch_size=batch_size,
        seed=dataset_seed,
        global_rank=global_rank,
        tokenizer=tokenizer  # Pass tokenizer to dataset
    )

    print(f"[RANK {global_rank}] Dataset created with tokenizer: {tokenizer}")

    # DataLoader for batched streaming
    # Use multiple workers to tokenize in parallel (hides CPU tokenization latency)
    # High worker count (plenty of CPU cores), low prefetch (avoid OOM from buffering)
    num_workers = args.num_workers

    # Worker initialization function to reseed each worker's PRNG
    # This ensures each worker reads from different file positions
    def worker_init_fn(worker_id):
        import numpy as np
        import random
        # Combine torch seed with worker_id for unique per-worker seed
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Set to None as the wrapper handles batching
        num_workers=num_workers,  # Many parallel tokenization workers
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch 4 batches per worker for smooth pipelining
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive to avoid respawn overhead
        worker_init_fn=worker_init_fn if num_workers > 0 else None  # Reseed each worker
    )

    if global_rank == 0 and num_workers > 0:
        print(f"Using {num_workers} DataLoader workers for parallel tokenization")

    # MEMORY CHECKPOINT 3: After DataLoader creation
    if global_rank == 0 and device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"[MEMORY] After DataLoader creation: {mem_allocated:.2f} GB")

    # --- 3. GOSSIP AND METRICS SETUP ---
    metrics_dir = os.path.join(checkpoint_dir, "metrics")
    metrics_log_path = os.path.join(metrics_dir, f"training_metrics_rank_{global_rank:03d}.tsv")
    with open(metrics_log_path, 'w') as f:
        header = [
            "rank", "step", "time_s", "loss", "val", 
            "tok_seen", "tok_sec", "lr",
            "docs", "gb_pos", "acc_steps", "opt", "grad_norm",
            # --- NEW COLUMNS ---
            "mix_out", "mix_in", "won", "lost", "tied", "failed"
        ]
        # Conditionally add the lock metric header
        if args.use_gossip_lock:
            header.append("locked")
        f.write('\t'.join(header) + '\n')

    # No probabilistic saving - using deterministic saving at fixed intervals instead
    save_callback = None  # Not used anymore, checkpoints saved deterministically in training loop

    # Configure gossip for DDP mode
    if args.ddp:
        # Only primary ranks participate in gossip
        if is_ddp_primary:
            # Calculate effective world size for gossip (number of nodes)
            num_nodes = world_size // ddp_world_size
            # Adjust node ID for gossip protocol
            gossip_rank = node_id
            gossip_world_size = num_nodes
        else:
            # Non-primary ranks don't participate in gossip
            gossip_rank = -1
            gossip_world_size = 0
    else:
        gossip_rank = global_rank
        gossip_world_size = world_size
        
    evolutionary_node = EvolutionaryTrainingNode(
        node_id=f"node_{global_rank}",
        model=base_model if args.ddp else model,  # Use base_model for DDP, model otherwise
        optimizer=optimizer,
        global_rank=gossip_rank if is_ddp_primary else -1,  # -1 disables gossip for non-primary
        local_rank=local_rank,
        world_size=gossip_world_size,
        data_parallel_rank=ddp_rank if args.ddp else global_rank,
        tp_size=1,
        mixing_probability=args.gossip_mixing_rate if is_ddp_primary else 0.0,
        output_dir=checkpoint_dir,
        merge_method=args.gossip_merge_method,
        recombination_alpha=args.gossip_recombination_alpha,
        optimizer_recombination=args.gossip_optimizer_recombination,
        gossip_temp_dir=args.gossip_temp_dir,
        fitness_window_size=args.gossip_fitness_window,
        use_node_local_lock=args.use_gossip_lock,
        use_filesystem_coordinator=args.filesystem_coordinator,
        save_callback=save_callback,
        data_path=args.data,
        chunk_size=chunk_size,
        batch_size=batch_size,
        p_value_threshold=args.gossip_p_value_threshold,
        validation_interval=args.validation_interval,
        validation_batches=args.validation_batches,
        gossip_lock_timeout=args.gossip_lock_timeout,
        tokenizer=tokenizer  # CRITICAL: Pass tokenizer so validation uses same tokenization as training!
    )
    
    # Only start gossip for primary ranks
    if is_ddp_primary:
        evolutionary_node.start_gossip_protocol()
    if global_rank == 0:
        proto_type = "Filesystem-Augmented" if args.filesystem_coordinator else "Pure TCP"
        print(f"\n{proto_type} Gossip protocol initialized and running.\n", flush=True)

    start_time = time.time()
    # Initialize per-GPU token counter
    total_tokens_processed = 0  # Now per-GPU, not global!
    total_tokens_since_reset = 0  # Track tokens for T/s calculation
    last_step_time = start_time  # Track time for it/s calculation
    # When resuming, skip warmup (already compiled in checkpoint)
    warmup_complete = resuming  # True if resuming, False if starting fresh
    bytes_at_reset = 0  # Track bytes processed at time of reset

    # --- 4. UNIFIED TRAINING LOOP ---
    
    # Modified log_metrics function  
    def log_metrics(step, train_loss, validation_fitness, mix_status, doc_stats, acc_steps, optimized, grad_norm=None, tokens_per_sec=0):
        nonlocal total_tokens_processed
        total_tokens_processed = doc_stats['bytes_processed']
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time  # Need elapsed for logging
        train_loss_scalar = train_loss.item() if torch.is_tensor(train_loss) else train_loss  # Sync only for logging

        values = [str(v) for v in [
            global_rank, step, f"{elapsed:.2f}", f"{train_loss_scalar:.6f}",
            f"{validation_fitness:.6f}" if validation_fitness != float('inf') else "NA",
            total_tokens_processed,  # Per-GPU tokens
            f"{tokens_per_sec:.2f}", f"{current_lr:.8f}",
            doc_stats['documents_processed'].item() if torch.is_tensor(doc_stats['documents_processed']) else doc_stats['documents_processed'],  # Sync only when logging
            f"{doc_stats['current_position'] / 1e9:.3f}",  # NEW
            acc_steps,  # NEW: accumulated steps
            1 if optimized else 0,  # NEW: whether optimizer stepped
            f"{grad_norm:.6f}" if grad_norm is not None else "NA",  # NEW: gradient norm
            mix_status['initiated_mixes'],
            mix_status['received_mixes'],
            mix_status['won_mixes'],
            mix_status['lost_mixes'],
            mix_status.get('tied_mixes', 0),  # NEW: tied mixes
            mix_status['failed_mixes']
        ]]
        # Conditionally add the lock metric value
        if args.use_gossip_lock:
            values.append(str(mix_status.get('skipped_due_to_lock', 0)))
        with open(metrics_log_path, 'a') as f:
            f.write('\t'.join(values) + '\n')

    # Main training loop (progress bar removed to save space for gradient logging)
    step = resume_step
    data_iterator = iter(train_loader)
    # Hidden states are now lists that will contain batched tensors
    hidden_state = []
    conv_buffers = []
    optimizer.zero_grad()
    
    # Track accumulated steps for gradient accumulation
    accumulated_steps = 0
    documents_processed_count = torch.tensor(0, device=device)  # Keep on GPU to avoid sync!

    # Z-gate logging setup (rank 0 only)
    # DISABLED: Z-gates are healthy, no need for monitoring
    z_stats_file = None
    # if global_rank == 0:
    #     z_stats_file = os.path.join(checkpoint_dir, f"z_stats_rank{global_rank}.tsv")
    #     print(f"[Rank {global_rank}] Logging z-gate stats to: {z_stats_file}")

    # MEMORY CHECKPOINT 4: Before training loop
    if global_rank == 0 and device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"[MEMORY] Before training loop: {mem_allocated:.2f} GB")

    while step < train_steps:
        # NOTE: Moved gossip updates to after optimization for safety
        # This prevents mid-batch model updates that could cause segfaults

        # === PROFILING: Start iteration ===
        if global_rank == 0 and step >= 20 and step < 120:
            prof_iter_start = prof_time()
            prof_step_times_global[step] = {}
            prof_data_start = prof_time()

        # Get a full batch of data
        chunk_data, is_doc_end, actual_lengths = next(data_iterator)

        if global_rank == 0 and step >= 20 and step < 120:
            prof_step_times_global[step]['data_load'] = prof_time() - prof_data_start

        # DEBUG: Check token range - DISABLED to avoid GPU sync stalls
        # The .item(), .tolist(), .min(), .max() calls all force CPU-GPU sync
        # if step < 5 and global_rank == 0:
        #     token_min = chunk_data.min().item()  # SYNC!
        #     token_max = chunk_data.max().item()  # SYNC!
        #     token_mean = chunk_data.float().mean().item()  # SYNC!
        #     print(f"[DEBUG STEP {step}] Token stats: min={token_min}, max={token_max}, mean={token_mean:.1f}", flush=True)
        #     print(f"[DEBUG STEP {step}] First 20 tokens: {chunk_data[0, :20].tolist()}", flush=True)  # SYNC!

        # Profiling disabled (was causing sync overhead)
        profile_this_step = False
        # if profile_this_step:
        #     t0 = time.time()

        chunk = chunk_data.to(device, non_blocking=True) # [B, SeqLen]
        is_doc_end = is_doc_end.to(device, non_blocking=True) # [B]
        # Move actual_lengths to GPU to prevent device mismatch
        actual_lengths = actual_lengths.to(device, non_blocking=True)

        # NO EXPLICIT SYNC: Let GPU kernel implicitly wait for data when needed
        # Explicit synchronize forces GPU to idle while waiting for transfer
        # With non_blocking=True + prefetch_factor=4, transfer overlaps with compute

        # MEMORY CHECKPOINT 6: After first batch loaded (only on step 0)
        if step == 0 and global_rank == 0 and device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[MEMORY] After first batch loaded to GPU: {mem_allocated:.2f} GB")

        # Acquire model mutex for entire forward/backward pass
        with evolutionary_node.model_mutex:
            # Mark that we're entering forward pass - no weight updates allowed
            evolutionary_node.enter_forward_pass(hidden_state, conv_buffers)

            if args.zero_order:
                # Zero-order optimization: Use CD-RGE optimizer with sequential batch scanning

                # Create batch provider for gradient accumulation
                # Each perturbation evaluates on grad_accum successive batches
                def batch_provider():
                    """
                    Fetch next batch from data stream and synchronize across GPUs.
                    Returns (chunk, actual_lengths, is_doc_end) tuple.
                    """
                    chunk_data, batch_is_doc_end, batch_actual_lengths = next(data_iterator)
                    batch_chunk = chunk_data.to(device, non_blocking=True)
                    batch_actual_lengths = batch_actual_lengths.to(device, non_blocking=True)
                    batch_is_doc_end = batch_is_doc_end.to(device, non_blocking=True)

                    # NO GPU SYNC! non_blocking=True handles async transfers properly.
                    # NO DATA BROADCAST! Each rank loads independently for data parallelism.
                    # DDP will synchronize gradients automatically during backward pass
                    # (only when accumulated_steps reaches grad_accum, not every step!)

                    return (batch_chunk, batch_actual_lengths, batch_is_doc_end)

                # Loss function for perturbation evaluation
                # Maintains hidden states across batches, resets at document boundaries
                def compute_loss_on_batch(batch_data, prev_hiddens=None, prev_conv=None):
                    """
                    Evaluate loss on a batch with hidden state tracking.

                    Args:
                        batch_data: (chunk, actual_lengths, is_doc_end) tuple
                        prev_hiddens: Hidden states from previous batch (or None for fresh start)
                        prev_conv: Conv buffers from previous batch (or None)

                    Returns:
                        (loss, next_hiddens, next_conv) tuple
                    """
                    batch_chunk, batch_actual_lengths, batch_is_doc_end = batch_data

                    # MeZO: NO autocast! It caches 16GB of gradient tensors we never use!
                    # Model is already in bf16, autocast just wastes memory.
                    result = model(
                        batch_chunk,
                        return_loss=True,
                        return_prev_hiddens=True,  # Need hidden states for next batch
                        prev_hiddens=prev_hiddens,
                        prev_conv_buffers=prev_conv,
                        actual_length=batch_actual_lengths
                    )

                    # Unpack result
                    if isinstance(result, tuple) and len(result) == 2:
                        loss, (next_hiddens, next_conv) = result
                    else:
                        loss = result
                        next_hiddens = None
                        next_conv = None

                    # Reset hidden states at document boundaries
                    if next_hiddens is not None and batch_is_doc_end is not None:
                        if isinstance(next_hiddens, list):
                            # Multi-layer: reset each layer
                            reset_mask = batch_is_doc_end.view(-1, 1)
                            next_hiddens = [h * (~reset_mask) for h in next_hiddens]
                        else:
                            # Single layer
                            if next_hiddens.dim() == 2:
                                reset_mask = batch_is_doc_end.view(-1, 1)
                            else:
                                reset_mask = batch_is_doc_end.view(-1, 1, 1)
                            next_hiddens = next_hiddens * (~reset_mask)

                    if next_conv is not None and batch_is_doc_end is not None:
                        conv_reset_mask = batch_is_doc_end.view(-1, 1, 1, 1)
                        next_conv = [b * (~conv_reset_mask) for b in next_conv]

                    return loss, next_hiddens, next_conv

                # Prepare first batch for evaluation
                first_batch = (chunk, actual_lengths, is_doc_end)

                # Create batch provider that yields first batch then subsequent batches
                batch_count = [0]  # Mutable counter for closure
                def batch_provider_with_first():
                    if batch_count[0] == 0:
                        batch_count[0] += 1
                        return first_batch
                    else:
                        return batch_provider()

                # MEMORY CHECKPOINT 5: Before first optimizer.step() (only on step 0)
                if step == 0 and global_rank == 0 and device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[MEMORY] Before first optimizer.step(): {mem_allocated:.2f} GB")

                # Call optimizer step with batch provider
                zo_result = optimizer.step(
                    loss_fn=compute_loss_on_batch,
                    batch_provider=batch_provider_with_first
                )
                chunk_loss = zo_result['loss']
                zo_grad_norm = zo_result.get('grad_norm', None)  # Extract gradient norm for logging

                # CRITICAL: Clear cache after optimizer.step() to free memory from 8 forward passes!
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # MEMORY CHECKPOINT: Log memory usage periodically (every 10 steps)
                if (step % 10 == 0) and global_rank == 0 and device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    mem_free = (torch.cuda.get_device_properties(device).total_memory / 1024**3) - mem_allocated
                    print(f"[MEM step{step:>4}] Allocated: {mem_allocated:5.2f} GB | Reserved: {mem_reserved:5.2f} GB | Free: {mem_free:5.2f} GB", flush=True)

                # Get hidden states from a single forward pass after optimization
                # Use return_loss=True for consistent return format
                # MeZO: NO autocast! Model already in bf16, no need to cache gradients.
                # MeZO: NO hidden state continuity! Each optimization step is independent (stateless).
                # MeZO: CRITICAL - torch.no_grad() to avoid caching 40+ GB of activations!
                with torch.no_grad():
                    result = model(
                        chunk,
                        return_loss=True,
                        return_prev_hiddens=True,
                        prev_hiddens=None,  # MeZO is stateless - no hidden state across steps!
                        prev_conv_buffers=None,  # No conv buffers either!
                        actual_length=actual_lengths
                    )

                # Unpack the same way as standard path
                if isinstance(result, tuple) and len(result) == 2:
                    _, (next_hidden_state, next_conv_buffers) = result
                else:
                    # Backward compatibility
                    next_hidden_state = None
                    next_conv_buffers = None

                # Check if optimizer actually updated parameters (gradient accumulation support)
                if zo_result.get('updated', True):  # True for backward compatibility with old optimizers
                    accumulated_steps = args.grad_accum  # Trigger optimization logic
                else:
                    accumulated_steps = 0  # Still accumulating, don't trigger yet
            else:
                # Standard backpropagation
                # Forward pass with both RNN hidden states and conv buffers

                # === CHUNKED cuDNN APPROACH: Reset hidden states BEFORE forward pass ===
                # Apply resets from previous chunk's document endings
                if hidden_state is not None and 'reset_next' in locals():
                    if reset_next.any():
                        # Reset hidden states for batch elements that ended docs in PREVIOUS chunk
                        # Handle both flat tensors and tuples (legacy EMA GRU format)
                        def masked_fill_hidden(h, mask):
                            if h is None:
                                return None
                            if isinstance(h, tuple):
                                return tuple(masked_fill_hidden(x, mask) for x in h)
                            # Expand mask to match hidden state dimensions (e.g., MultiHeadElman: batch, nheads, headdim)
                            m = mask
                            while m.dim() < h.dim():
                                m = m.unsqueeze(-1)
                            return h.masked_fill(m, 0.0)

                        mask = reset_next.unsqueeze(-1)
                        hidden_state = [masked_fill_hidden(h, mask) for h in hidden_state]

                # Save is_doc_end for next chunk's reset
                reset_next = is_doc_end  # [B] bool - will be used BEFORE next chunk

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16 and not getattr(args, 'no_autocast', False)):
                    # No TBPTT mode: reset hidden state each chunk for fair comparison with Mamba2
                    effective_hidden = None if getattr(args, 'no_tbptt', False) else hidden_state
                    effective_conv = None if getattr(args, 'no_tbptt', False) else conv_buffers
                    result = model(
                        chunk,
                        return_loss=True,
                        return_prev_hiddens=True,
                        prev_hiddens=effective_hidden,
                        prev_conv_buffers=effective_conv,
                        actual_length=actual_lengths,
                        doc_boundaries=None  # Not needed - resets handled above
                    )

                # Unpack the result - could be just loss or loss + (hiddens, buffers)
                if isinstance(result, tuple) and len(result) == 2:
                    loss, (next_hidden_state, next_conv_buffers) = result
                else:
                    # Backward compatibility - model without conv buffers
                    loss = result
                    next_hidden_state = None
                    next_conv_buffers = None

                # Scale loss for gradient accumulation
                scaled_loss = loss / args.grad_accum
                chunk_loss = loss.detach()  # Keep on GPU, convert to scalar only when logging

                # Track accumulation progress
                accumulated_steps += 1

                # Backward pass with conditional DDP sync
                # Only sync gradients on the final accumulation step
                should_sync_now = (accumulated_steps >= args.grad_accum)

                # FlashRNN workaround: scale loss to avoid bias gradient kernel bug
                backward_loss = scaled_loss * flashrnn_loss_scale

                if args.ddp and not should_sync_now:
                    # Skip gradient sync during accumulation (steps 1-15 of 16)
                    with model.no_sync():
                        backward_loss.backward()
                else:
                    # Allow DDP sync on final step (step 16 of 16), or always if not DDP
                    # FlashRNN workaround: Sync CUDA before DDP gradient sync
                    if flashrnn_loss_scale != 1.0 and args.ddp:
                        torch.cuda.synchronize()
                    backward_loss.backward()
                    # FlashRNN workaround: Ensure all CUDA work complete after backward
                    if flashrnn_loss_scale != 1.0:
                        torch.cuda.synchronize()

            # Mark that we've exited forward pass - safe for weight updates
            evolutionary_node.exit_forward_pass()

        # Profile timing breakdown (DISABLED)
        # if profile_this_step:
        #     torch.cuda.synchronize()
        #     t_fwd_bwd = time.time() - t0
        #     print(f"[PROFILE STEP {step}] Forward+Backward: {t_fwd_bwd*1000:.1f}ms", flush=True)
        
        # --- KEY LOGIC: DYNAMIC HIDDEN STATE HANDLING (NO ALLOCATION!) ---
        # Count documents processed in main process
        # Defer doc count sync - keep on GPU
        documents_processed_count += is_doc_end.sum()  # Accumulate on GPU

        # Hidden states are now reset IN-PLACE during forward pass (no allocation needed!)
        # Just detach and pass through for next iteration
        # Handle both flat tensors and tuples (e.g., EMA GRU returns (h_fast, h_slow))
        if next_hidden_state and next_hidden_state[0] is not None:
            def detach_hidden(h):
                if isinstance(h, tuple):
                    return tuple(x.detach() for x in h)
                return h.detach()
            hidden_state = [detach_hidden(h) for h in next_hidden_state]
        else:
            hidden_state = []  # No hidden state (e.g., CausalConvGRU)

        # Conv buffers still need reset at document boundaries (not handled by GRU in-place logic)
        if next_conv_buffers and len(next_conv_buffers) > 0 and isinstance(next_conv_buffers[0], torch.Tensor):
            conv_reset_mask = is_doc_end.view(-1, 1, 1)
            conv_buffers = [torch.where(conv_reset_mask, torch.zeros_like(b), b.detach())
                           for b in next_conv_buffers]
        else:
            conv_buffers = [] # Ensure it's an empty list if no conv
        
        # Use a fixed, synchronous optimization schedule
        should_optimize = accumulated_steps >= args.grad_accum

        # Calculate gradient norm before optimization (for logging)
        grad_norm = None
        if should_optimize:
            if args.zero_order:
                # Zero-order: Extract gradient norm from optimizer result
                # Optimizer step already happened in the forward pass
                grad_norm = zo_grad_norm  # Use the gradient norm computed by MeZO optimizer
                accumulated_steps = 0
            else:
                # Standard backpropagation: handle gradients
                # FlashRNN workaround: Sync CUDA streams before gradient handling
                # This ensures all FlashRNN JIT kernels are complete before NCCL
                if flashrnn_loss_scale != 1.0:
                    torch.cuda.synchronize()
                    inv_scale = 1.0 / flashrnn_loss_scale
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(inv_scale)

                # Calculate gradient norm for logging - SINGLE SYNC instead of per-param
                # Use torch.nn.utils.clip_grad_norm_ which computes norm efficiently on GPU
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if grads:
                    # Stack all grads into a single tensor, compute norm, single .item() call
                    total_norm_tensor = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
                    grad_norm = total_norm_tensor.item()  # Single GPU sync!
                else:
                    grad_norm = 0.0

                # Conditionally perform gradient clipping if args.grad_clip is set > 0
                if args.grad_clip > 0.0:
                    # Clip the gradients using the value from the command line
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                # The optimizer step proceeds as usual, operating on the (now clipped) gradients
                optimizer.step()

                optimizer.zero_grad()
                accumulated_steps = 0
            
            # Log z-gate statistics on rank 0 after every gradient update
            # DISABLED: Z-gates are healthy, no need for monitoring
            # if global_rank == 0 and z_stats_file is not None:
            #     try:
            #         # Log first 4 layers by default (adjust as needed)
            #         log_z_stats_to_tsv(model, chunk_data, hidden_state, step, z_stats_file, num_layers_to_log=4)
            #     except Exception as e:
            #         print(f"[Rank {global_rank}] Warning: Failed to log z-stats: {e}")

            # NO SYNC: DDP already syncs gradients at accumulation boundary
            # Explicit sync forces GPU to idle every 16 steps, dropping utilization to 30%

            # SAFE GOSSIP SYNCHRONIZATION POINT - all updates happen here
            # First check and apply any pending model updates
            was_updated, needs_optimizer_reset = evolutionary_node.apply_pending_update()
            if was_updated:
                if global_rank == 0:
                    print(f"Rank {global_rank} received model update at step {step}")
                
                # DDP: Broadcast updated weights to other ranks in DDP group
                if args.ddp and is_ddp_primary:
                    for param in model.parameters():
                        dist.broadcast(param.data, src=ddp_rank, group=ddp_group)
                    if global_rank == 0:
                        print(f"Node {node_id}: Broadcasted gossip update to DDP group")
                
                # Try to restore cached hidden states if available
                cached_hidden, cached_conv = evolutionary_node.get_cached_hidden_states()
                if cached_hidden is not None:
                    hidden_state = cached_hidden
                    conv_buffers = cached_conv if cached_conv else []
                    if global_rank == 0:
                        print(f"Rank {global_rank} restored hidden states after model update")
                else:
                    # CRITICAL: Reset hidden states if no cache available
                    hidden_state = []
                    conv_buffers = []
                
                if needs_optimizer_reset:
                    if global_rank == 0:
                        print(f"Rank {global_rank} resetting optimizer at step {step}")

                    # Recreate optimizer based on training mode
                    if args.zero_order:
                        from zero_order_optimizer import CD_RGE_Optimizer
                        epsilon = args.zo_epsilon if args.zo_epsilon is not None else args.lr
                        optimizer = CD_RGE_Optimizer(
                            model=model,
                            learning_rate=args.lr,
                            epsilon=epsilon,
                            n_perturbations=args.zo_n_perturbations,
                            world_size=world_size,
                            rank=global_rank,
                            chunk_size=args.zo_memory_chunk,
                            grad_accum=args.grad_accum
                        )
                    else:
                        optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay) if args.schedulefree else AdamW(model.parameters(), lr=args.lr, betas=(args.sf_beta, args.sf_beta2), weight_decay=args.weight_decay)
                        if args.schedulefree: optimizer.train()
                
                # DDP: Synchronize after updates
                if args.ddp:
                    dist.barrier(group=ddp_group)
                    evolutionary_node.optimizer = optimizer
                    total_actual_tokens = 0  # Reset token counter
            
            # Now safe to update fitness and request mixing
            evolutionary_node.update_fitness(chunk_loss, step)
            evolutionary_node.check_for_updates()
            evolutionary_node.request_mix()
            
            current_validation_fitness = evolutionary_node.get_current_fitness()
        else:
            evolutionary_node.update_fitness(chunk_loss, step)
            current_validation_fitness = evolutionary_node.get_current_fitness()
        
        # Get status and log with document stats
        status = evolutionary_node.get_status()
        # Aggregate stats across all streams for accurate reporting
        all_stats = train_dataset.get_all_stats()
        doc_stats = {
            'documents_processed': documents_processed_count,  # FIXED: Count in main process
            'bytes_processed': sum(s['bytes_processed'] for s in all_stats) if all_stats else 0,
            'tokens_processed': sum(s.get('tokens_processed', 0) for s in all_stats) if all_stats else 0,
            'file_wraps': sum(s['file_wraps'] for s in all_stats) if all_stats else 0,
            'current_position': 0  # Not meaningful with multiple streams
        }
        # First calculate timing metrics (needed for log_metrics)
        # Reset timing after step 2 (after torch.compile warmup) - ALL RANKS
        if step == 2 and not warmup_complete:
            start_time = time.time()
            last_step_time = start_time
            total_tokens_since_reset = 0  # Reset token counter
            warmup_complete = True
            if global_rank == 0:
                print("\n=== Timing reset after torch.compile warmup ===")
        
        # Calculate timing metrics - ALL RANKS
        current_time = time.time()
        elapsed = current_time - start_time
        step_time = current_time - last_step_time
        last_step_time = current_time
        
        # Calculate it/s (always show, it's useful even during warmup)
        iterations_per_sec = 1.0 / step_time if step_time > 0 else 0
        
        # Track DATA tokens for this step (tokens that contribute to learning)
        # This is batch_size * chunk_size * world_size regardless of method
        # (Zero-order does more forward passes, but same data throughput)
        tokens_this_step = chunk_size * batch_size * world_size
        total_tokens_since_reset += tokens_this_step

        # Calculate tok/s per-step (more useful than cumulative average)
        # Uses step_time instead of cumulative elapsed to show current throughput
        if warmup_complete and step_time > 0.1:  # Only show after warmup, require >0.1s step time
            tokens_per_sec = tokens_this_step / step_time
        else:
            tokens_per_sec = 0  # Don't show during warmup or for very fast steps
        
        log_metrics(step, chunk_loss, current_validation_fitness, status, doc_stats, accumulated_steps, should_optimize, grad_norm, tokens_per_sec)
        
        if global_rank == 0:
            
            # Console logging with it/s added
            doc_count = doc_stats['documents_processed'].item() if torch.is_tensor(doc_stats['documents_processed']) else doc_stats['documents_processed']
            loss_scalar = chunk_loss.item() if torch.is_tensor(chunk_loss) else chunk_loss  # Sync only for logging (rank 0 only)
            if grad_norm is not None:
                log_str = f"Step {step:6d}: L={loss_scalar:.4f} V={status['fitness']:.4f} G={grad_norm:.4f} T/s={tokens_per_sec:.0f} it/s={iterations_per_sec:.2f} D={doc_count}"
            else:
                log_str = f"Step {step:6d}: L={loss_scalar:.4f} V={status['fitness']:.4f} G={'NA':>6s} T/s={tokens_per_sec:.0f} it/s={iterations_per_sec:.2f} D={doc_count}"
            if 'skipped_due_to_lock' in status:
                log_str += f" skipped={status['skipped_due_to_lock']}"
            print(log_str)

        # Background checkpoint saving: copy to CPU (fast), then save in thread (slow)
        if step > 0 and args.save_every > 0 and step % args.save_every == 0:
            if global_rank == 0:
                print(f"Rank 0: Copying checkpoint to CPU at step {step}, loss {chunk_loss:.4f}")

                # Copy state dicts to CPU (fast, ~1-2s)
                checkpoint_data_cpu = {
                    'step': step,
                    'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'optimizer_state_dict': {
                        k: {k2: v2.cpu() if torch.is_tensor(v2) else v2 for k2, v2 in v.items()}
                        if isinstance(v, dict) else v
                        for k, v in optimizer.state_dict().items()
                    },
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'training_loss': chunk_loss,
                    'validation_fitness': current_validation_fitness,
                    'model_config': model_config
                }

                # Launch background thread to save (slow disk I/O doesn't block training!)
                def cleanup():
                    update_symlinks_and_cleanup(checkpoint_dir, args.keep_checkpoints,
                                                args.keep_elite, args.milestone_every)

                save_thread = threading.Thread(
                    target=save_checkpoint_background,
                    args=(checkpoint_data_cpu, checkpoint_dir, step, global_rank, chunk_loss, cleanup)
                )
                save_thread.daemon = True
                save_thread.start()

                print(f"Rank 0: Checkpoint copy complete, saving in background...")

            # Quick barrier just to sync that all ranks are ready (doesn't wait for disk I/O!)
            if args.ddp:
                dist.barrier()

        # === PROFILING: End iteration + Report === (DISABLED)
        # if global_rank == 0 and step >= 20 and step < 120:
        #     prof_iter_end = prof_time()
        #     total_time = prof_iter_end - prof_iter_start
        #     times = prof_step_times_global[step]
        #
        #     # Calculate data load time
        #     data_load = times.get('data_load', 0)
        #
        #     # Categorize step type
        #     is_opt_step = (accumulated_steps == 0)  # Just did optimizer
        #     category = "OPT" if is_opt_step else "REG"
        #
        #     print(f"[PROF{step:4d} {category}] "
        #           f"Data={data_load*1000:4.0f}ms "
        #           f"Total={total_time*1000:6.0f}ms "
        #           f"it/s={1/total_time:.2f}",
        #           flush=True)
        #
        #     if step == 119:
        #         # Print summary statistics
        #         import numpy as np
        #         print("\n" + "="*80)
        #         print("PROFILING SUMMARY (steps 20-119):")
        #         print("="*80)
        #
        #         # Analyze variance
        #         all_times = []
        #         opt_times = []
        #         reg_times = []
        #
        #         for s in range(20, 120):
        #             if s in prof_step_times_global:
        #                 if s not in prof_step_times_global:
        #                     continue
        #                 step_start = prof_iter_start  # This is wrong but approximate
        #                 # Actually we can't calculate this easily, skip detailed analysis
        #                 pass
        #
        #         print("✅ Profiling complete. Check [PROF] lines above for per-step timing.")
        #         print("="*80)

        step += 1

    # pbar.close()  # No progress bar anymore
    if args.filesystem_coordinator:
        evolutionary_node.stop_gossip_protocol()
    # checkpoint_manager.stop()  # DISABLED - no background thread

    # Save final checkpoint if not already saved at this step
    final_step = step - 1  # The last completed step
    if global_rank == 0 and args.save_every > 0 and final_step % args.save_every != 0:
        print(f"Rank 0: Saving final checkpoint at step {final_step}, loss {chunk_loss:.4f}")

        checkpoint_data_cpu = {
            'step': final_step,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': {
                k: {k2: v2.cpu() if torch.is_tensor(v2) else v2 for k2, v2 in v.items()}
                if isinstance(v, dict) else v
                for k, v in optimizer.state_dict().items()
            },
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'training_loss': chunk_loss,
            'validation_fitness': current_validation_fitness,
            'model_config': model_config
        }

        # Save synchronously since training is complete
        save_checkpoint_atomic(checkpoint_data_cpu, checkpoint_dir, final_step, global_rank, chunk_loss)
        update_symlinks_and_cleanup(checkpoint_dir, args.keep_checkpoints,
                                    args.keep_elite, args.milestone_every)
        print(f"Rank 0: Final checkpoint saved.")

    if global_rank == 0: print("\nTraining complete.")

if __name__ == "__main__":
    main()

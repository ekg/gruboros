import random
import torch
import time
import os
import logging
import socket
import threading
import queue
import struct
import glob
import re
import json
import numpy as np
from scipy import stats
import statistics
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
from .validation_tracker import ValidationTracker
from .network_utils import NetworkUtils
from .structured_logger import GossipLogger
from .filesystem_coordinator import FilesystemCoordinator
import tempfile
import uuid
import fcntl
import contextlib
import gc
import psutil

def get_memory_log_path(rank: int):
    """Get the memory log file path for this rank"""
    # Try to use the output directory from environment or fall back to current dir
    output_dir = os.environ.get('GRUBOROS_OUTPUT_DIR', '.')
    os.makedirs(os.path.join(output_dir, 'memory_logs'), exist_ok=True)
    return os.path.join(output_dir, 'memory_logs', f'memory_rank_{rank}.log')

def log_memory_usage(rank: int, message: str):
    """Logs current memory usage to a dedicated file"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    log_message = f"[{timestamp}] {message}: Process RSS: {rss_gb:.2f} GB, Node Available: {available_gb:.2f} GB\n"
    
    # Write to memory log file
    log_path = get_memory_log_path(rank)
    with open(log_path, 'a') as f:
        f.write(log_message)

def check_memory_headroom(rank: int, required_bytes: int, message: str) -> bool:
    """Check if there's enough memory headroom for an operation"""
    safety_buffer_bytes = 4 * 1024**3  # 4 GB safety buffer
    total_required = required_bytes + safety_buffer_bytes
    
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    required_gb = total_required / (1024**3)
    
    if available_gb < required_gb:
        # Log warning to memory log file
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] MEMORY WARNING - {message}: Need {required_gb:.2f} GB, only {available_gb:.2f} GB available\n"
        log_path = get_memory_log_path(rank)
        with open(log_path, 'a') as f:
            f.write(log_message)
        return False
    return True

@contextlib.contextmanager
def file_lock(lock_path, timeout=10.0):
    lock_file = None
    try:
        lock_file = open(lock_path, 'a')
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
                return
            except BlockingIOError:
                time.sleep(0.05)
        raise TimeoutError(f"Could not acquire lock on {lock_path} within {timeout}s")
    finally:
        if lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

@dataclass
class WeightUpdate:
    payload_path: str
    source_node: str
    source_ema_loss: float
    correlation_id: str

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 global_rank: int, local_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01, 
                 output_dir: Optional[str] = None,
                 merge_method: str = 'clonal',
                 recombination_alpha: float = 0.5,
                 optimizer_recombination: str = 'reset',
                 gossip_temp_dir: Optional[str] = None,
                 fitness_window_size: int = 1000,
                 use_node_local_lock: bool = False,
                 use_filesystem_coordinator: bool = False,
                 save_callback: Optional[callable] = None,
                 data_path: str = None,
                 chunk_size: int = None,
                 p_value_threshold: float = 0.01,
                 validation_interval: int = 10000,
                 validation_sequences: int = 32,
                 validation_sequence_length: int = 8192):
        
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        self.mixing_probability = mixing_probability
        self.merge_method = merge_method
        self.recombination_alpha = recombination_alpha
        self.optimizer_recombination = optimizer_recombination
        self.use_node_local_lock = use_node_local_lock
        self.use_filesystem_coordinator = use_filesystem_coordinator
        self.output_dir = output_dir
        self.save_callback = save_callback
        self.p_value_threshold = p_value_threshold
        
        self.coordinator: Optional[FilesystemCoordinator] = None
        if self.use_filesystem_coordinator:
            self.coordinator = FilesystemCoordinator(
                global_rank=self.global_rank, world_size=self.world_size, output_dir=output_dir
            )
            self.coordinator.set_fitness_provider(self.get_current_fitness)
        
        self.mixing_rng = random.Random(42 + self.global_rank * 1000)
        self.incoming_updates = queue.Queue()
        self.step_notifications = queue.Queue()
        
        # Replace fitness_tracker with validation_tracker
        self.validation_tracker = ValidationTracker(
            data_path=data_path,
            chunk_size=chunk_size,
            validation_interval=validation_interval,
            num_sequences=validation_sequences,
            sequence_length=validation_sequence_length,
            window_size=10
        )
        self.validation_lock = threading.Lock()
        self.current_step = 0
        
        self.pending_update = None
        self.weight_stream = torch.cuda.Stream()
        self.weight_ready_event = torch.cuda.Event()
        self.weight_transfer_lock = threading.Lock()
        
        self.logger = GossipLogger(self.node_id, self.global_rank, self.local_rank, self.data_parallel_rank, output_dir)
        
        if gossip_temp_dir:
            self.gossip_temp_dir = gossip_temp_dir
        else:
            self.gossip_temp_dir = os.environ.get('LUSTRE_SCRATCH') or os.environ.get('SCRATCH') or tempfile.gettempdir()
        
        if self.global_rank == 0:
            os.makedirs(self.gossip_temp_dir, exist_ok=True)
            print(f"Gossip temporary directory set to: {self.gossip_temp_dir}")
        
        std_logger = logging.getLogger(f'evolutionary_node_{self.node_id}')
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(self.global_rank, self.local_rank, std_logger)
        
        self.mixing_attempts = 0
        self.successful_mixes = 0
        self.current_step = 0
        
        if self.use_node_local_lock:
            self.node_lock_path = os.path.join(self.gossip_temp_dir, "gossip.node.lock")
            if self.local_rank == 0:
                try: os.remove(self.node_lock_path)
                except FileNotFoundError: pass
                Path(self.node_lock_path).touch()
            if self.world_size > 1: time.sleep(0.5)
            self.logger.log_event("NODE_LOCK_ENABLED", message=f"Lock file: {self.node_lock_path}")

        self.outbound_mixes_attempted = 0
        self.inbound_mixes_attempted = 0
        self.mixes_won = 0
        self.mixes_lost = 0
        self.mixes_skipped_locked = 0
        self.peer_list: Dict[str, dict] = {}
        
        self.logger.log_event("NODE_STARTUP", message=f"TP_size={self.tp_size}, mixing_prob={self.mixing_probability}")

    def update_fitness(self, loss_value: float, step: int):
        """Called every step, but only validates periodically"""
        self.current_step = step
        
        # Check if we should run validation
        if self.validation_tracker.should_validate(step):
            with self.validation_lock:
                fitness = self.validation_tracker.run_validation(
                    self.model, 
                    step, 
                    seed=self.global_rank * 1000
                )
                
            # Log validation event
            self.logger.log_event(
                "VALIDATION_UPDATE",
                step=step,
                fitness=fitness,
                message=f"Validated on {self.validation_tracker.num_sequences} sequences of {self.validation_tracker.sequence_length} bytes"
            )
        
        # Notify gossip thread about step
        try:
            self.step_notifications.put_nowait(step)
        except queue.Full:
            pass

    def get_current_fitness(self) -> float:
        """Return validation-based fitness"""
        with self.validation_lock:
            return self.validation_tracker.get_fitness()

    def check_for_updates(self) -> Optional[WeightUpdate]:
        """Check for incoming weight updates and start transfer"""
        try:
            update = self.incoming_updates.get_nowait()
            # We now process immediately instead of staging
            was_applied, needs_reset = self._start_async_weight_transfer(update)
            if was_applied:
                return update
            return None
        except queue.Empty:
            return None

    def _start_async_weight_transfer(self, update: WeightUpdate) -> tuple[bool, bool]:
        if self.use_node_local_lock:
            try:
                with file_lock(self.node_lock_path, timeout=15.0):
                    return self._load_and_transfer_payload(update)
            except TimeoutError:
                self.logger.log_event("UPDATE_SKIPPED_LOCK", message="Node lock busy, skipping weight update application.")
                if os.path.exists(update.payload_path): os.remove(update.payload_path)
                return False, False
        else:
            return self._load_and_transfer_payload(update)

    def _load_and_transfer_payload(self, update: WeightUpdate):
        """Load weights using PyTorch's native memory mapping to minimize memory usage"""
        with self.weight_transfer_lock:
            try:
                log_memory_usage(self.global_rank, "Before mmap load")
                
                device = next(self.model.parameters()).device
                
                # Step 1: Memory-map the checkpoint (no actual loading yet!)
                mmap_checkpoint = torch.load(
                    update.payload_path,
                    map_location='cpu',
                    mmap=True,  # This is the key!
                    weights_only=False  # We need optimizer state too
                )
                
                # Extract metadata without loading tensors
                source_ema_loss = mmap_checkpoint.get('ema_loss', float('inf'))
                
                # Step 2: Process weights one at a time to minimize memory
                if self.merge_method == 'recombination':
                    needs_optimizer_reset = self._apply_recombination_mmap(
                        mmap_checkpoint, device, self.recombination_alpha
                    )
                else:  # clonal replacement
                    needs_optimizer_reset = self._apply_clonal_mmap(
                        mmap_checkpoint, device
                    )
                
                # Update fitness tracker
                with self.validation_lock:
                    self.validation_tracker.inherit_fitness(source_ema_loss)
                
                # Clean up - mmap checkpoint will be released
                del mmap_checkpoint
                gc.collect()
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Remove the payload file
                os.remove(update.payload_path)
                
                self.successful_mixes += 1
                self.logger.log_event(
                    "MMAP_UPDATE_APPLIED",
                    step=self.current_step,
                    correlation_id=update.correlation_id,
                    message=f"Memory-mapped update completed"
                )
                
                log_memory_usage(self.global_rank, "After mmap update")
                
                return True, needs_optimizer_reset
                
            except Exception as e:
                self.logger.log_event("MMAP_LOAD_ERROR", message=str(e))
                if os.path.exists(update.payload_path):
                    os.remove(update.payload_path)
                return False, False

    def _apply_recombination_mmap(self, mmap_checkpoint, device, alpha):
        """Apply recombination using memory-mapped weights"""
        with torch.no_grad():
            # Get the state dict (still memory-mapped, not loaded!)
            mmap_state_dict = mmap_checkpoint['model_state_dict']
            
            # Process each parameter individually
            for name, param in self.model.named_parameters():
                if name in mmap_state_dict:
                    # This is when the tensor actually gets loaded from disk
                    incoming_tensor = mmap_state_dict[name]
                    
                    # Since mmap tensors are read-only, we must copy to device
                    # This loads only this parameter into memory
                    incoming_device = incoming_tensor.to(device, non_blocking=True)
                    
                    # Blend directly into existing parameter
                    param.mul_(1.0 - alpha).add_(incoming_device, alpha=alpha)
                    
                    # Immediately free the device copy
                    del incoming_device
                    
                    # Force CUDA to release memory
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Synchronize to ensure all transfers complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Handle optimizer state if needed
        if self.optimizer_recombination == 'interpolate' and 'optimizer_state_dict' in mmap_checkpoint:
            return self._apply_optimizer_recombination_mmap(
                mmap_checkpoint['optimizer_state_dict'], device, alpha
            )
        elif self.optimizer_recombination == 'reset':
            return True  # Needs reset
        
        return False

    def _apply_clonal_mmap(self, mmap_checkpoint, device):
        """Apply clonal replacement using memory-mapped weights"""
        with torch.no_grad():
            mmap_state_dict = mmap_checkpoint['model_state_dict']
            
            # Copy each parameter individually
            for name, param in self.model.named_parameters():
                if name in mmap_state_dict:
                    # Load from mmap and copy directly to parameter
                    incoming_tensor = mmap_state_dict[name]
                    param.copy_(incoming_tensor.to(device, non_blocking=True))
                    
                    # Clear CUDA cache after each parameter
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in mmap_checkpoint:
            self._load_optimizer_state_mmap(
                mmap_checkpoint['optimizer_state_dict'], device
            )
        
        return False  # No reset needed

    def _apply_optimizer_recombination_mmap(self, mmap_opt_state, device, alpha):
        """Blend optimizer states using memory mapping"""
        with torch.no_grad():
            incoming_state = mmap_opt_state['state']
            
            for param_idx, param_state in self.optimizer.state.items():
                if param_idx in incoming_state:
                    incoming_param_state = incoming_state[param_idx]
                    
                    # Blend momentum buffers
                    for buffer_name in ['exp_avg', 'exp_avg_sq']:
                        if buffer_name in param_state and buffer_name in incoming_param_state:
                            # Load only this buffer from mmap
                            incoming_buffer = incoming_param_state[buffer_name].to(
                                device, non_blocking=True
                            )
                            
                            # Blend in-place
                            param_state[buffer_name].mul_(1.0 - alpha).add_(
                                incoming_buffer, alpha=alpha
                            )
                            
                            del incoming_buffer
                            
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
        
        return False  # No reset needed

    def _load_optimizer_state_mmap(self, mmap_opt_state, device):
        """Load optimizer state from memory-mapped checkpoint"""
        with torch.no_grad():
            # Update param groups
            if 'param_groups' in mmap_opt_state:
                for i, group in enumerate(mmap_opt_state['param_groups']):
                    if i < len(self.optimizer.param_groups):
                        for key in ['lr', 'betas', 'eps', 'weight_decay']:
                            if key in group:
                                self.optimizer.param_groups[i][key] = group[key]
            
            # Update state tensors
            incoming_state = mmap_opt_state['state']
            
            for param_idx in self.optimizer.state:
                if param_idx in incoming_state:
                    current_param_state = self.optimizer.state[param_idx]
                    incoming_param_state = incoming_state[param_idx]
                    
                    # Copy tensor states one at a time
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in incoming_param_state and key in current_param_state:
                            mmap_tensor = incoming_param_state[key]
                            current_param_state[key].copy_(
                                mmap_tensor.to(device, non_blocking=True)
                            )
                            
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    
                    # Copy scalar states
                    for key in ['step']:
                        if key in incoming_param_state:
                            current_param_state[key] = incoming_param_state[key]

    def _apply_large_tensor_recombination(self, param, mmap_tensor, device, alpha, 
                                         chunk_size_mb=100):
        """Mix very large tensors in chunks to minimize memory peaks"""
        
        # Calculate chunk size in elements
        bytes_per_element = param.element_size()
        chunk_elements = (chunk_size_mb * 1024 * 1024) // bytes_per_element
        
        param_flat = param.view(-1)
        total_elements = param_flat.numel()
        
        # Process in chunks
        for start_idx in range(0, total_elements, chunk_elements):
            end_idx = min(start_idx + chunk_elements, total_elements)
            
            # Load only this chunk from mmap
            chunk_slice = slice(start_idx, end_idx)
            mmap_chunk = mmap_tensor.view(-1)[chunk_slice]
            
            # Copy chunk to device
            device_chunk = mmap_chunk.to(device, non_blocking=True)
            
            # Mix in-place
            param_flat[chunk_slice].mul_(1.0 - alpha).add_(device_chunk, alpha=alpha)
            
            # Free chunk immediately
            del device_chunk
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Ensure param maintains its original shape
        param.data = param_flat.view(param.shape)

    def log_mmap_stats(self, phase: str, mmap_checkpoint):
        """Log memory mapping statistics"""
        if hasattr(mmap_checkpoint, '_mmap_info'):
            # PyTorch might expose mmap info in future versions
            self.logger.log_event(
                "MMAP_STATS",
                message=f"{phase}: Using memory-mapped checkpoint"
            )
        
        # Log current memory state
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.log_event(
                    "GPU_MEMORY",
                    message=f"{phase} - GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

    def apply_pending_update(self) -> tuple[bool, bool]:
        """Apply any pending weight updates"""
        try:
            update = self.incoming_updates.get_nowait()
            return self._start_async_weight_transfer(update)
        except queue.Empty:
            return False, False

    def start_gossip_protocol(self):
        if self.coordinator: self.coordinator.start()
        self.gossip_running = True
        self.gossip_thread = threading.Thread(target=self._gossip_worker, daemon=True); self.gossip_thread.start()
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True); self.server_thread.start()
        self.logger.log_event("GOSSIP_PROTOCOL_STARTED", message="Background threads launched")

    def _gossip_worker(self):
        time.sleep(2)
        for bootstrap_addr in self.bootstrap_nodes: self.peer_list[bootstrap_addr] = {}
        self.logger.log_event("PEER_DISCOVERY", message=f"Found {len(self.peer_list)} peers.")
        while self.gossip_running:
            try:
                step = self.step_notifications.get(timeout=1.0); self.current_step = step
                if self.mixing_rng.random() < self.mixing_probability:
                    if self.use_node_local_lock:
                        try:
                            with file_lock(self.node_lock_path, timeout=0.05): self._try_mix_with_peer()
                        except TimeoutError: self.mixes_skipped_locked += 1
                    else: self._try_mix_with_peer()
                self.step_notifications.task_done()
            except queue.Empty: continue
            except Exception as e: self.logger.log_event("GOSSIP_WORKER_ERROR", message=str(e)); time.sleep(1)

    def _server_loop(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); server_sock.settimeout(1.0); NetworkUtils.optimize_socket(server_sock)
        try:
            server_sock.bind(('0.0.0.0', NetworkUtils.get_gossip_port(self.local_rank))); server_sock.listen(5); self.logger.log_event("SERVER_STARTED", message=f"Listening on {self.logger.node_identity}")
            while self.gossip_running:
                try:
                    client_sock, addr = server_sock.accept()
                    threading.Thread(target=self._do_handle_request, args=(client_sock, addr), daemon=True).start()
                except socket.timeout: continue
                except Exception as e:
                    if self.gossip_running: self.logger.log_event("SERVER_ERROR", message=str(e)); time.sleep(1)
        finally: server_sock.close()
    
    def _do_handle_request(self, client_sock, addr):
        """Handle incoming gossip request with statistical validation"""
        peer_addr = f"{addr[0]}:{addr[1]}"
        correlation_id = self.logger.generate_correlation_id()
        self.inbound_mixes_attempted += 1
        
        try:
            NetworkUtils.optimize_socket(client_sock)
            client_sock.settimeout(30.0)
            
            # 1. Generate validation seed
            seed = int(time.time() * 1000) % (2**32)
            
            # 2. Send validation challenge
            challenge = json.dumps({
                'seed': seed,
                'correlation_id': correlation_id
            })
            client_sock.send(f"VALIDATE:{challenge}".encode())
            
            # 3. Wait for peer's validation results
            peer_data = b''
            while len(peer_data) < 4:
                peer_data += client_sock.recv(4 - len(peer_data))
            
            result_size = struct.unpack('!I', peer_data)[0]
            peer_results_data = b''
            while len(peer_results_data) < result_size:
                peer_results_data += client_sock.recv(result_size - len(peer_results_data))
            
            peer_losses = np.array(json.loads(peer_results_data.decode()))
            
            # 4. Run our validation with same seed
            our_losses = self.validation_tracker.evaluate_for_gossip(self.model, seed)
            
            # Update our fitness with this validation result
            our_mean = np.mean(our_losses)
            with self.validation_lock:
                self.validation_tracker.validation_losses.append(our_mean)
                self.validation_tracker.current_fitness = statistics.median(self.validation_tracker.validation_losses)
            
            # Log this fitness update
            self.logger.log_event(
                "VALIDATION_UPDATE",
                step=self.current_step,
                fitness=self.validation_tracker.current_fitness,
                message=f"Updated from gossip challenge (mean: {our_mean:.4f})"
            )
            
            # 5. Send our results
            our_results = json.dumps(our_losses.tolist()).encode()
            client_sock.send(struct.pack('!I', len(our_results)))
            client_sock.send(our_results)
            
            # 6. Statistical test
            t_stat, p_value = stats.ttest_rel(our_losses, peer_losses)
            our_mean = np.mean(our_losses)
            peer_mean = np.mean(peer_losses)
            
            self.logger.log_event(
                "VALIDATION_COMPARISON",
                step=self.current_step,
                correlation_id=correlation_id,
                peer_addr=peer_addr,
                message=f"p={p_value:.4f}, our_mean={our_mean:.4f}, peer_mean={peer_mean:.4f}, t={t_stat:.3f}"
            )
            
            # 7. Make decision based on p-value
            if p_value < self.p_value_threshold:  # Statistically significant difference
                if our_mean < peer_mean:
                    # We win
                    self.mixes_won += 1
                    send_optimizer = self.optimizer_recombination == 'interpolate'
                    client_sock.send(f"WINNER:SENDING_WEIGHTS:{send_optimizer}".encode())
                    client_sock.settimeout(120.0)
                    self._send_our_weights_to_peer(client_sock, correlation_id, send_optimizer)
                    
                    # Opportunistic save
                    if self.save_callback:
                        try:
                            self.save_callback(self.current_step, our_mean, opportunistic=True)
                        except Exception as e:
                            self.logger.log_event("OPPORTUNISTIC_SAVE_FAILED", message=str(e))
                else:
                    # We lose
                    self.mixes_lost += 1
                    client_sock.send(f"LOSER:SEND_ME_WEIGHTS:{self.optimizer_recombination}".encode())
                    client_sock.settimeout(120.0)
                    payload_path, source_fitness = self._receive_weights_from_peer(client_sock, correlation_id)
                    
                    if payload_path:
                        update = WeightUpdate(
                            payload_path=payload_path,
                            source_node=peer_addr,
                            source_ema_loss=source_fitness,
                            correlation_id=correlation_id
                        )
                        self.incoming_updates.put(update)
                        self.logger.log_event(
                            "WEIGHT_UPDATE_QUEUED",
                            step=self.current_step,
                            correlation_id=correlation_id,
                            peer_addr=peer_addr,
                            fitness=source_fitness
                        )
            else:
                # No significant difference
                client_sock.send(b"NO_MIX:NO_SIGNIFICANT_DIFFERENCE")
                self.logger.log_event(
                    "NO_SIGNIFICANT_DIFFERENCE",
                    step=self.current_step,
                    correlation_id=correlation_id,
                    peer_addr=peer_addr,
                    message=f"p={p_value:.3f} > 0.05"
                )
                
        except Exception as e:
            self.logger.log_event(
                "INCOMING_REQUEST_ERROR",
                step=self.current_step,
                correlation_id=correlation_id,
                peer_addr=peer_addr,
                message=str(e)
            )
        finally:
            client_sock.close()

    def _try_mix_with_peer(self):
        """Initiate mixing with validation-based comparison"""
        if not self.peer_list:
            return
            
        self.outbound_mixes_attempted += 1
        correlation_id = self.logger.generate_correlation_id()
        
        # Select random peer
        local_identity = self.logger.node_identity
        other_peers = [p for p in self.peer_list.keys() if p != local_identity]
        if not other_peers:
            return
            
        peer_address = self.mixing_rng.choice(other_peers)
        
        self.logger.log_event(
            "MIX_ATTEMPT_START",
            step=self.current_step,
            correlation_id=correlation_id,
            peer_addr=peer_address,
            fitness=self.get_current_fitness()
        )
        
        host, port = peer_address.split(':')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        NetworkUtils.optimize_socket(sock)
        
        try:
            sock.settimeout(10.0)
            sock.connect((host, int(port)))
            
            # Receive validation challenge
            challenge_data = sock.recv(4096).decode()
            if not challenge_data.startswith("VALIDATE:"):
                return
                
            challenge = json.loads(challenge_data[9:])
            seed = challenge['seed']
            
            # Run validation with provided seed
            our_losses = self.validation_tracker.evaluate_for_gossip(self.model, seed)
            
            # Update our fitness with this validation result
            our_mean = np.mean(our_losses)
            with self.validation_lock:
                self.validation_tracker.validation_losses.append(our_mean)
                self.validation_tracker.current_fitness = statistics.median(self.validation_tracker.validation_losses)
            
            # Log this fitness update
            self.logger.log_event(
                "VALIDATION_UPDATE", 
                step=self.current_step,
                fitness=self.validation_tracker.current_fitness,
                message=f"Updated from gossip challenge (mean: {our_mean:.4f})"
            )
            
            # Send our results
            our_results = json.dumps(our_losses.tolist()).encode()
            sock.send(struct.pack('!I', len(our_results)))
            sock.send(our_results)
            
            # Receive peer results
            peer_data = b''
            while len(peer_data) < 4:
                peer_data += sock.recv(4 - len(peer_data))
                
            result_size = struct.unpack('!I', peer_data)[0]
            peer_results_data = b''
            while len(peer_results_data) < result_size:
                peer_results_data += sock.recv(result_size - len(peer_results_data))
                
            peer_losses = np.array(json.loads(peer_results_data.decode()))
            
            # Wait for decision
            sock.settimeout(30.0)
            decision = sock.recv(1024).decode()
            
            if decision.startswith("WINNER:SENDING_WEIGHTS"):
                # They won, we receive
                self.mixes_lost += 1
                sock.settimeout(120.0)
                payload_path, source_fitness = self._receive_weights_from_peer(sock, correlation_id)
                if payload_path:
                    self.incoming_updates.put(WeightUpdate(
                        payload_path=payload_path,
                        source_node=peer_address,
                        source_ema_loss=source_fitness,
                        correlation_id=correlation_id
                    ))
                    
            elif decision.startswith("LOSER:SEND_ME_WEIGHTS"):
                # We won, send weights
                self.mixes_won += 1
                loser_policy = decision.split(':')[-1] if ':' in decision else 'reset'
                send_optimizer = loser_policy == 'interpolate'
                sock.settimeout(120.0)
                self._send_our_weights_to_peer(sock, correlation_id, send_optimizer)
                
                # Opportunistic save
                if self.save_callback:
                    try:
                        fitness = self.get_current_fitness()
                        self.save_callback(self.current_step, fitness, opportunistic=True)
                    except Exception as e:
                        self.logger.log_event("OPPORTUNISTIC_SAVE_FAILED", message=str(e))
                        
            elif decision.startswith("NO_MIX"):
                # No significant difference
                self.logger.log_event(
                    "NO_MIX_PEER_DECISION",
                    step=self.current_step,
                    correlation_id=correlation_id,
                    peer_addr=peer_address
                )
                
            self.mixing_attempts += 1
            
        except Exception as e:
            self.logger.log_event(
                "MIX_ATTEMPT_FAILED",
                step=self.current_step,
                correlation_id=correlation_id,
                peer_addr=peer_address,
                message=str(e)
            )
        finally:
            sock.close()

    def _send_our_weights_to_peer(self, sock, correlation_id: str, send_optimizer_state: bool = True):
        """Save weights in a format compatible with memory mapping"""
        start_time = time.time()
        temp_filename = os.path.join(self.gossip_temp_dir, f"gossip_payload_{uuid.uuid4()}.pt")
        
        try:
            log_memory_usage(self.global_rank, "Before creating payload")
            
            # Build checkpoint dict - this is still in memory unfortunately
            # But we'll save it in a way that supports mmap loading
            checkpoint = {
                'validation_fitness': self.get_current_fitness(),
                'ema_loss': self.get_current_fitness(),  # Keep for compatibility
                'model_state_dict': {
                    name: param.detach().cpu() 
                    for name, param in self.model.named_parameters()
                }
            }
            
            if send_optimizer_state:
                opt_state = self.optimizer.state_dict()
                checkpoint['optimizer_state_dict'] = {
                    'param_groups': opt_state['param_groups'],
                    'state': {
                        param_id: {
                            k: v.cpu() if torch.is_tensor(v) else v
                            for k, v in param_state.items()
                        }
                        for param_id, param_state in opt_state['state'].items()
                    }
                }
                log_memory_usage(self.global_rank, "After creating optimizer payload")
            else:
                log_memory_usage(self.global_rank, "Skipping optimizer state (not requested)")
            
            # Save with settings that enable memory mapping
            torch.save(
                checkpoint, 
                temp_filename,
                _use_new_zipfile_serialization=True  # Required for mmap!
            )
            
            # Stream file to peer
            file_size = os.path.getsize(temp_filename)
            sock.send(struct.pack('!Q', file_size))
            
            with open(temp_filename, 'rb') as f:
                while chunk := f.read(4 * 1024 * 1024):
                    sock.sendall(chunk)
            
            self.logger.log_event(
                "WEIGHT_SEND_COMPLETE",
                step=self.current_step,
                correlation_id=correlation_id,
                data_size_bytes=file_size,
                transfer_time_ms=(time.time() - start_time) * 1000,
                fitness=checkpoint['validation_fitness'],
                message=f"Optimizer included: {send_optimizer_state}"
            )
            
        except Exception as e:
            self.logger.log_event("WEIGHT_SEND_ERROR", step=self.current_step, correlation_id=correlation_id, message=str(e))
        finally:
            if os.path.exists(temp_filename): os.remove(temp_filename)

    def _receive_weights_from_peer(self, sock, correlation_id: str) -> tuple[Optional[str], Optional[float]]:
        start_time = time.time(); temp_filename = os.path.join(self.gossip_temp_dir, f"gossip_payload_{uuid.uuid4()}.pt")
        try:
            length_data = b'';
            while len(length_data) < 8: length_data += sock.recv(8 - len(length_data))
            expected_size = struct.unpack('!Q', length_data)[0]; bytes_received = 0
            
            if not check_memory_headroom(self.global_rank, expected_size, "payload download"):
                self.logger.log_event("PAYLOAD_DOWNLOAD_SKIPPED", message="Insufficient memory for payload download")
                return None, None
            
            with open(temp_filename, 'wb') as f:
                while bytes_received < expected_size: chunk = sock.recv(min(4096*1024, expected_size - bytes_received)); f.write(chunk); bytes_received += len(chunk)
            loaded_data = torch.load(temp_filename, map_location='cpu'); source_ema_loss = loaded_data.get('ema_loss', float('inf'))
            self.logger.log_event("WEIGHT_RECEIVE_COMPLETE", step=self.current_step, correlation_id=correlation_id, data_size_bytes=expected_size, transfer_time_ms=(time.time() - start_time) * 1000, fitness=source_ema_loss)
            return temp_filename, source_ema_loss
        except Exception as e:
            self.logger.log_event("WEIGHT_RECEIVE_ERROR", step=self.current_step, correlation_id=correlation_id, message=str(e))
            if os.path.exists(temp_filename): os.remove(temp_filename)
            return None, None

    def stop_gossip_protocol(self):
        if self.coordinator: self.coordinator.stop()
        self.gossip_running = False
        if self.gossip_thread: self.gossip_thread.join(timeout=5)
        if self.server_thread: self.server_thread.join(timeout=5)
        self.logger.log_event("GOSSIP_PROTOCOL_STOPPED", message="Gossip protocol stopped.")

    def request_mix(self): pass

    def get_status(self) -> dict:
        outbound = self.outbound_mixes_attempted; inbound = self.inbound_mixes_attempted; won = self.mixes_won; lost = self.mixes_lost
        failed_mixes = (outbound + inbound) - (won + lost)
        status = {'node_id': self.node_id, 'fitness': self.get_current_fitness(), 'peer_count': len(self.peer_list), 'successful_mixes': self.successful_mixes, 'current_step': self.current_step, 'initiated_mixes': outbound, 'received_mixes': inbound, 'won_mixes': won, 'lost_mixes': lost, 'failed_mixes': failed_mixes}
        if self.use_node_local_lock: status['skipped_due_to_lock'] = self.mixes_skipped_locked
        return status

    def attempt_rejuvenation(self, model: torch.nn.Module, alpha: float, tiebreaker_threshold: float = 0.005) -> tuple[bool, bool]:
        elite_symlinks = glob.glob(os.path.join(self.output_dir, "elite_*.pt"))
        if not elite_symlinks: return False, False
        try:
            my_current_fitness = self.get_current_fitness()
            if my_current_fitness == float('inf'): return False, False
            best_candidate_checkpoint = None; chosen_symlink_path = None; best_elite_fitness = my_current_fitness; best_elite_step = -1
            for symlink_path in elite_symlinks:
                try:
                    target_path = os.path.basename(os.readlink(symlink_path)); loss_match = re.search(r'loss_([\d.inf]+)', target_path); step_match = re.search(r'step_(\d+)', target_path)
                    if not loss_match or not step_match: continue
                    elite_fitness = float(loss_match.group(1)); elite_step = int(step_match.group(1))
                    if elite_fitness >= my_current_fitness: continue
                    if elite_fitness < best_elite_fitness * (1.0 - tiebreaker_threshold):
                        best_elite_fitness = elite_fitness; best_elite_step = elite_step; chosen_symlink_path = symlink_path
                    elif elite_fitness < best_elite_fitness:
                        if elite_step > best_elite_step:
                            best_elite_fitness = elite_fitness; best_elite_step = elite_step; chosen_symlink_path = symlink_path
                except (FileNotFoundError, ValueError, IndexError, OSError): continue
            if chosen_symlink_path: best_candidate_checkpoint = torch.load(chosen_symlink_path, map_location='cpu')
            else:
                self.logger.log_event("REJUVENATION_SKIPPED", step=self.current_step, message=f"No elite model found with fitness < {my_current_fitness:.4f}", fitness=my_current_fitness)
                return False, False
            with torch.no_grad():
                elite_state_dict = best_candidate_checkpoint['model_state_dict']
                for name, param in model.named_parameters():
                    if name in elite_state_dict:
                        elite_param = elite_state_dict[name].to(param.device, non_blocking=True)
                        param.mul_(1.0 - alpha).add_(elite_param, alpha=alpha)
            needs_reset = True
            if 'ema_fitness' in best_candidate_checkpoint: self.fitness_tracker.inherit_fitness(best_candidate_checkpoint['ema_fitness'])
            self.logger.log_event("REJUVENATION_SUCCESS", step=self.current_step, message=f"Blended with {os.path.basename(chosen_symlink_path)} (L:{best_elite_fitness:.4f} S:{best_elite_step})", fitness=best_candidate_checkpoint.get('ema_fitness'))
            return True, needs_reset
        except Exception as e:
            self.logger.log_event("REJUVENATION_FAILED", step=self.current_step, message=str(e))
            return False, False
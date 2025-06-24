import random
import torch
import io
import time
import os
import logging
import socket
import threading
import queue
import struct
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import torch.distributed as dist
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils
from .structured_logger import GossipLogger
import tempfile
import uuid
import fcntl
import contextlib

@contextlib.contextmanager
def file_lock(lock_path, timeout=0.1):
    """
    A non-blocking file-based lock for inter-process synchronization on a single node.
    If the lock cannot be acquired within the timeout, it raises a TimeoutError.
    This is a self-contained copy for modularity.
    """
    lock_file = open(lock_path, 'w')
    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
                return
            except BlockingIOError:
                time.sleep(0.01)
        raise TimeoutError(f"Could not acquire lock on {lock_path} within {timeout}s")
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

@dataclass
class WeightUpdate:
    state_dict: dict
    optimizer_state_dict: Optional[dict]  # Added optimizer state transfer
    source_node: str
    source_ema_loss: float  # Added EMA loss transfer
    correlation_id: str    # Added correlation tracking

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 global_rank: int, local_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01, 
                 output_dir: Optional[str] = None,
                 merge_method: str = 'clonal',
                 recombination_alpha: float = 0.5,
                 optimizer_recombination: str = 'reset',
                 gossip_temp_dir: Optional[str] = None,
                 fitness_decay_factor: float = 0.95,
                 use_node_local_lock: bool = False):
        # Store parameters as instance variables FIRST
        self.node_id = node_id
        self.model = model  # Main thread owns this
        self.optimizer = optimizer  # Store optimizer reference
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
        
        # Now initialize other attributes
        self.mixing_rng = random.Random(42 + self.global_rank * 1000)
        
        # Thread-safe communication
        self.incoming_updates = queue.Queue()  # Background thread -> Main thread
        self.step_notifications = queue.Queue()  # Main thread -> Background thread
        self.fitness_tracker = FitnessTracker(decay_factor=fitness_decay_factor)
        self.fitness_lock = threading.Lock()
        
        # NEW: Add async weight management
        self.pending_update = None
        self.weight_stream = torch.cuda.Stream()  # Dedicated stream for weight ops
        self.weight_ready_event = torch.cuda.Event()  # Synchronization event
        self.weight_transfer_lock = threading.Lock()
        
        # Replace standard logger with structured gossip logger
        self.logger = GossipLogger(self.node_id, self.global_rank, self.local_rank, self.data_parallel_rank, output_dir)
        
        # Configure and verify the temporary directory
        if gossip_temp_dir:
            self.gossip_temp_dir = gossip_temp_dir
        else:
            # Sensible default for HPC systems, falling back to standard /tmp
            self.gossip_temp_dir = os.environ.get('LUSTRE_SCRATCH') or os.environ.get('SCRATCH') or tempfile.gettempdir()
        
        # Rank 0 creates the directory, others wait. This avoids race conditions.
        if self.global_rank == 0:
            os.makedirs(self.gossip_temp_dir, exist_ok=True)
            print(f"Gossip temporary directory set to: {self.gossip_temp_dir}")
        if self.world_size > 1:
            # Import dist here to avoid circular imports
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        
        # Pre-allocate pinned memory buffers for common model sizes
        self._setup_pinned_buffers()
        
        # UPDATE the port calculation to use the new simple method
        self.gossip_port = NetworkUtils.get_gossip_port(self.local_rank)
        
        # Thread control
        self.gossip_running = False
        self.gossip_thread = None
        self.server_thread = None
    
    def _setup_pinned_buffers(self):
        """Pre-allocate pinned memory for faster transfers"""
        # Estimate model size for buffer allocation
        model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        # Allocate pinned buffer (with some headroom)
        buffer_size = int(model_size_bytes * 1.2)
        self.pinned_buffer = torch.empty(buffer_size, dtype=torch.uint8, pin_memory=True)
        print(f"Allocated {buffer_size / 1e6:.1f}MB pinned buffer for weight transfers")
        
        # Create standard logger for bootstrap discovery
        std_logger = logging.getLogger(f'evolutionary_node_{self.node_id}')
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(
            self.global_rank, self.local_rank, std_logger
        )
        
        self.mixing_attempts = 0
        self.successful_mixes = 0
        self.current_step = 0
        
        # Conditional node-local lock setup
        if self.use_node_local_lock:
            self.node_lock_path = os.path.join(self.gossip_temp_dir, "gossip.node.lock")
            # The first rank on each node initializes/clears the lock file.
            # `global_rank == local_rank` is a safe way to identify the first rank on a node.
            if self.global_rank == self.local_rank:
                 if os.path.exists(self.node_lock_path):
                     os.remove(self.node_lock_path)
                 Path(self.node_lock_path).touch()
            # All ranks must wait for the file to be created before proceeding.
            if self.world_size > 1 and dist.is_initialized():
                dist.barrier()
            self.logger.log_event("NODE_LOCK_ENABLED", message=f"Lock file: {self.node_lock_path}")

        # New gossip metrics
        self.outbound_mixes_attempted = 0
        self.inbound_mixes_attempted = 0
        self.mixes_won = 0  # When we send our weights
        self.mixes_lost = 0  # When we receive weights
        if self.use_node_local_lock:
            self.mixes_skipped_locked = 0
        
        # --- Initialize peer_list in the constructor to prevent race conditions ---
        # The gossip worker thread populates this, but the main thread may access it before population.
        self.peer_list: Dict[str, dict] = {}
        
        # Log node startup
        self.logger.log_event("NODE_STARTUP", 
                             message=f"TP_size={self.tp_size}, mixing_prob={self.mixing_probability}")
        
    def update_fitness(self, loss_value: float, step: int):
        """Called by main thread after each training step"""
        with self.fitness_lock:
            old_fitness = self.fitness_tracker.get_fitness()
            self.fitness_tracker.update(loss_value)
            new_fitness = self.fitness_tracker.get_fitness()
        
        # Fitness logging removed to prevent periodic stalls
        
        # Notify gossip worker about this step
        try:
            self.step_notifications.put_nowait(step)
        except queue.Full:
            # If queue is full, skip this notification
            pass
    
    def get_current_fitness(self) -> float:
        """Thread-safe fitness access"""
        with self.fitness_lock:
            return self.fitness_tracker.get_fitness()
    
    def check_for_updates(self) -> Optional[WeightUpdate]:
        """Non-blocking check for weight updates"""
        try:
            update = self.incoming_updates.get_nowait()
            # Start async transfer immediately
            self._start_async_weight_transfer(update)
            return update
        except queue.Empty:
            return None
    
    def _start_async_weight_transfer(self, update: WeightUpdate):
        """Start transferring weights on background CUDA stream"""
        with self.weight_transfer_lock:
            # Move to background stream for async transfer
            with torch.cuda.stream(self.weight_stream):
                device = next(self.model.parameters()).device
                
                # Pin memory and transfer asynchronously
                for name, param in update.state_dict.items():
                    if not param.is_pinned():
                        param = param.pin_memory()
                    # Non-blocking transfer to GPU
                    update.state_dict[name] = param.to(device, non_blocking=True)
                
                # Record event when transfer is complete
                self.weight_ready_event.record(self.weight_stream)
            
            # Store for later application
            self.pending_update = update
    
    def apply_pending_update(self) -> tuple[bool, bool]:
        """
        Apply any pending weight update.
        Returns:
            Tuple[bool, bool]: (was_update_applied, needs_optimizer_reset)
        """
        if self.pending_update is None:
            return False, False

        with self.weight_transfer_lock:
            # Wait for async transfer to complete
            torch.cuda.current_stream().wait_event(self.weight_ready_event)

            start_time = time.time()
            needs_optimizer_reset = False
            
            # Core recombination logic
            if self.merge_method == 'recombination':
                winner_model_state = self.pending_update.state_dict
                alpha = self.recombination_alpha

                # 1. Interpolate model parameters
                with torch.no_grad():
                    for name, loser_param in self.model.named_parameters():
                        if name in winner_model_state and loser_param.dtype.is_floating_point:
                            winner_param = winner_model_state[name].to(loser_param.device)
                            loser_param.mul_(1.0 - alpha).add_(winner_param, alpha=alpha)

                # 2. Handle optimizer state based on strategy
                if self.optimizer_recombination == 'interpolate' and self.pending_update.optimizer_state_dict:
                    # Interpolate the optimizer state
                    winner_optim_state = self.pending_update.optimizer_state_dict['state']
                    loser_optim_state = self.optimizer.state

                    with torch.no_grad():
                        for p_id, winner_p_state in winner_optim_state.items():
                            if p_id in loser_optim_state:
                                loser_p_state = loser_optim_state[p_id]
                                # Interpolate 'exp_avg' and 'exp_avg_sq'
                                for key in ['exp_avg', 'exp_avg_sq']:
                                    if key in loser_p_state and key in winner_p_state:
                                        loser_tensor = loser_p_state[key]
                                        winner_tensor = winner_p_state[key].to(loser_tensor.device)
                                        loser_tensor.mul_(1.0 - alpha).add_(winner_tensor, alpha=alpha)

                elif self.optimizer_recombination == 'reset':
                    # Signal the main loop to reset the optimizer
                    needs_optimizer_reset = True
            
            else:  # clonal
                # Overwrite model and optimizer state completely
                self.model.load_state_dict(self.pending_update.state_dict)
                if self.pending_update.optimizer_state_dict:
                    # Move optimizer state tensors to the correct device before loading
                    winner_optim_state = self.pending_update.optimizer_state_dict
                    device = next(self.model.parameters()).device
                    for state in winner_optim_state['state'].values():
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                state[key] = value.to(device)
                    self.optimizer.load_state_dict(winner_optim_state)

            load_time = (time.time() - start_time) * 1000

            # Inherit fitness
            with self.fitness_lock:
                self.fitness_tracker.inherit_fitness(self.pending_update.source_ema_loss)
            
            self.successful_mixes += 1
            
            self.pending_update = None
            return True, needs_optimizer_reset
    
    def apply_update(self, update: WeightUpdate):
        """Legacy method - now just queues for async processing"""
        # This gets called by the old training loop, but now it's non-blocking
        pass  # Weight transfer already started in check_for_updates()

    def start_gossip_protocol(self):
        """Start the background gossip thread"""
        self.gossip_running = True
        self.gossip_thread = threading.Thread(target=self._gossip_worker, daemon=True)
        self.gossip_thread.start()
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        self.logger.log_event("GOSSIP_PROTOCOL_STARTED", 
                             message="Background threads launched")
    
    def _gossip_worker(self):
        """Background thread that handles gossip networking - NOW STEP-BASED"""
        # Wait a bit for server to start
        time.sleep(2)
        
        # Discover and populate peers (the attribute is already initialized)
        for bootstrap_addr in self.bootstrap_nodes:
            self.peer_list[bootstrap_addr] = {}
        
        self.logger.log_event("PEER_DISCOVERY", 
                             message=f"Found {len(self.peer_list)} peers: {list(self.peer_list.keys())}")
        
        # Main gossip loop - NOW STEP-BASED
        while self.gossip_running:
            try:
                # Wait for step notification from main thread
                # Block with timeout so we can still check gossip_running
                try:
                    step = self.step_notifications.get(timeout=1.0)
                    self.current_step = step
                    
                    # Check if we should attempt mixing on this step
                    if self.mixing_rng.random() < self.mixing_probability:
                        # Conditional locking logic
                        if self.use_node_local_lock:
                            try:
                                with file_lock(self.node_lock_path, timeout=0.05):
                                    self.logger.log_event("MIX_DECISION_LOCKED", 
                                                         step=step,
                                                         fitness=self.get_current_fitness(),
                                                         message="Acquired node lock, proceeding with mix.")
                                    self._try_mix_with_peer()
                            except TimeoutError:
                                self.mixes_skipped_locked += 1
                                self.logger.log_event("MIX_ATTEMPT_SKIPPED", 
                                                     step=step,
                                                     message="Node lock was busy, skipping mix attempt.")
                        else:
                            # Original behavior: mix without locking
                            self.logger.log_event("MIX_DECISION", 
                                                 step=step,
                                                 fitness=self.get_current_fitness(),
                                                 message="Attempting mix (locking disabled).")
                            self._try_mix_with_peer()
                    
                    # Mark step notification as processed
                    self.step_notifications.task_done()
                    
                except queue.Empty:
                    # Timeout - just continue the loop to check gossip_running
                    continue
                    
            except Exception as e:
                self.logger.log_event("GOSSIP_WORKER_ERROR", 
                                     message=str(e))
                time.sleep(1)
    
    def _server_loop(self):
        """Background thread: listen for incoming gossip requests"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)
        
        # Apply optimizations
        NetworkUtils.optimize_socket(server_sock)
        
        try:
            server_sock.bind(('0.0.0.0', self.gossip_port))
            server_sock.listen(5)
            self.logger.log_event("SERVER_STARTED", 
                                 message=f"Listening on {self.logger.node_identity}")
            
            while self.gossip_running:
                try:
                    client_sock, addr = server_sock.accept()
                    # Handle each connection in separate thread
                    threading.Thread(
                        target=self._handle_incoming_request,
                        args=(client_sock, addr),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue  # Check self.gossip_running again
                except Exception as e:
                    if self.gossip_running:
                        self.logger.log_event("SERVER_ERROR", 
                                             message=str(e))
                        time.sleep(1)
                    
        finally:
            server_sock.close()
    
    def _handle_incoming_request(self, client_sock, addr):
        """Background thread: handle one incoming gossip request"""
        peer_addr = f"{addr[0]}:{addr[1]}"
        correlation_id = None
        
        # Count inbound attempts
        self.inbound_mixes_attempted += 1
        
        try:
            NetworkUtils.optimize_socket(client_sock)
            client_sock.settimeout(5.0)  # Short timeout for initial fitness exchange
            
            # Receive fitness comparison with correlation ID
            data = client_sock.recv(1024).decode()
            if not data.startswith("EMA_LOSS:"):
                return
            
            # Parse: EMA_LOSS:2.345678:CID:abc12345
            parts = data.split(":")
            if len(parts) >= 4 and parts[2] == "CID":
                peer_fitness = float(parts[1])
                correlation_id = parts[3]
            else:
                peer_fitness = float(parts[1])
                correlation_id = self.logger.generate_correlation_id()
                
            our_fitness = self.get_current_fitness()
            
            self.logger.log_event("INCOMING_FITNESS_COMPARISON", 
                                 step=self.current_step,
                                 fitness=our_fitness,
                                 correlation_id=correlation_id,
                                 peer_addr=peer_addr,
                                 message=f"peer_fitness={peer_fitness:.4f}")
            
            if our_fitness < peer_fitness:
                # We win - send our weights
                self.mixes_won += 1
                client_sock.send(b"SENDING_WEIGHTS")
                # Extend timeout for weight transfer
                client_sock.settimeout(120.0)
                self._send_our_weights_to_peer(client_sock, correlation_id)
                
            else:
                # We lose - receive their weights
                self.mixes_lost += 1
                client_sock.send(b"SEND_ME_WEIGHTS")
                # Extend timeout for weight transfer
                client_sock.settimeout(120.0)
                new_weights, new_optimizer_state, source_fitness, source_ema_loss = self._receive_weights_from_peer(client_sock, correlation_id)
                
                if new_weights:
                    # Queue the update for main thread (don't apply here!)
                    update = WeightUpdate(
                        state_dict=new_weights,
                        optimizer_state_dict=new_optimizer_state,
                        source_node=peer_addr,
                        source_ema_loss=source_ema_loss,
                        correlation_id=correlation_id
                    )
                    self.incoming_updates.put(update)
                    self.logger.log_event("WEIGHT_UPDATE_QUEUED", 
                                         step=self.current_step,
                                         correlation_id=correlation_id,
                                         peer_addr=peer_addr,
                                         fitness=source_ema_loss)
                
        except Exception as e:
            self.logger.log_event("INCOMING_REQUEST_ERROR", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 peer_addr=peer_addr,
                                 message=str(e))
        finally:
            client_sock.close()
    
    def _try_mix_with_peer(self):
        """Enhanced mixing with correlation tracking"""
        if not self.peer_list:
            return
            
        # Count outbound attempts
        self.outbound_mixes_attempted += 1
        correlation_id = self.logger.generate_correlation_id()
        
        # Choose random peer - use our actual node identity
        local_identity = self.logger.node_identity
        other_peers = [p for p in self.peer_list.keys() if p != local_identity]
        if not other_peers:
            return
            
        peer_address = self.mixing_rng.choice(other_peers)
        
        self.logger.log_event("MIX_ATTEMPT_START", 
                             step=self.current_step,
                             correlation_id=correlation_id,
                             peer_addr=peer_address,
                             fitness=self.get_current_fitness())
        
        host, port_str = peer_address.split(':')
        port = int(port_str)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        NetworkUtils.optimize_socket(sock)
        
        try:
            sock.settimeout(3.0)  # Short timeout for initial connection
            sock.connect((host, port))
            
            # Send our fitness with correlation ID
            our_fitness = self.get_current_fitness()
            message = f"EMA_LOSS:{our_fitness:.6f}:CID:{correlation_id}".encode()
            sock.send(message)
            
            # Keep short timeout for fitness comparison response
            sock.settimeout(2.0)
            response = sock.recv(1024)
            
            # Extend timeout for weight transfers
            sock.settimeout(120.0)
            
            if response == b"SENDING_WEIGHTS":
                # Peer is sending, so we lost this exchange
                self.mixes_lost += 1
                new_weights, new_optimizer_state, source_fitness, source_ema_loss = self._receive_weights_from_peer(sock, correlation_id)
                if new_weights:
                    update = WeightUpdate(
                        state_dict=new_weights,
                        optimizer_state_dict=new_optimizer_state,
                        source_node=peer_address,
                        source_ema_loss=source_ema_loss,
                        correlation_id=correlation_id
                    )
                    self.incoming_updates.put(update)
                    
            elif response == b"SEND_ME_WEIGHTS":
                # We are sending, so we won this exchange
                self.mixes_won += 1
                self._send_our_weights_to_peer(sock, correlation_id)
                
            self.mixing_attempts += 1
            
        except Exception as e:
            self.logger.log_event("MIX_ATTEMPT_FAILED", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 peer_addr=peer_address,
                                 message=str(e))
        finally:
            sock.close()
    
    def _send_our_weights_to_peer(self, sock, correlation_id: str):
        """Send weights AND optimizer state by streaming from a temporary file to avoid OOM."""
        start_time = time.time()
        
        temp_filename = os.path.join(self.gossip_temp_dir, f"gossip_payload_{uuid.uuid4()}.pt")
        
        try:
            # Get model state and move to CPU
            model_cpu_state = {name: param.cpu() for name, param in self.model.state_dict().items()}
            
            # Get optimizer state and move its tensors to CPU
            optimizer_cpu_state = {}
            if hasattr(self.optimizer, 'state_dict'):
                opt_state_dict = self.optimizer.state_dict()
                optimizer_cpu_state = {
                    'state': {},
                    'param_groups': opt_state_dict['param_groups']
                }
                for param_id, param_state in opt_state_dict['state'].items():
                    optimizer_cpu_state['state'][param_id] = {}
                    for key, value in param_state.items():
                        if torch.is_tensor(value):
                            optimizer_cpu_state['state'][param_id][key] = value.cpu()
                        else:
                            optimizer_cpu_state['state'][param_id][key] = value
            
            # Include EMA loss in the transfer
            our_ema_loss = self.get_current_fitness()
            transfer_data = {
                'model_state_dict': model_cpu_state,
                'optimizer_state_dict': optimizer_cpu_state,
                'ema_loss': our_ema_loss,
                'correlation_id': correlation_id
            }
            
            # Save directly to the temporary file on the configured filesystem
            torch.save(transfer_data, temp_filename, _use_new_zipfile_serialization=True)
            
            # Get the file size to send as a prefix
            file_size = os.path.getsize(temp_filename)
            sock.send(struct.pack('!Q', file_size))
            
            # Stream the file chunk-by-chunk to keep RAM usage low
            with open(temp_filename, 'rb') as f:
                while True:
                    chunk = f.read(4 * 1024 * 1024)  # Read in 4MB chunks
                    if not chunk:
                        break  # End of file
                    sock.sendall(chunk)
            
            transfer_time = (time.time() - start_time) * 1000
            throughput = file_size / 1e6 / (transfer_time / 1000) if transfer_time > 0 else 0
            
            self.logger.log_event("WEIGHT_SEND_COMPLETE",
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 data_size_bytes=file_size,
                                 transfer_time_ms=transfer_time,
                                 fitness=our_ema_loss,
                                 message=f"Throughput: {throughput:.2f} MB/s")
            
        except Exception as e:
            self.logger.log_event("WEIGHT_SEND_ERROR", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 message=str(e))
        finally:
            # Robustly clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError as e:
                    self.logger.log_event("TEMP_FILE_CLEANUP_ERROR", message=f"Failed to remove {temp_filename}: {e}")
    
    def _receive_weights_from_peer(self, sock, correlation_id: str) -> tuple[Optional[dict], Optional[dict], Optional[float], Optional[float]]:
        """Receive weights and optimizer state by streaming to a temporary file."""
        start_time = time.time()
        
        temp_filename = os.path.join(self.gossip_temp_dir, f"gossip_payload_{uuid.uuid4()}.pt")

        try:
            # Receive 8-byte length prefix
            length_data = b''
            while len(length_data) < 8:
                chunk = sock.recv(8 - len(length_data))
                if not chunk:
                    return None, None, None, None
                length_data += chunk
            
            expected_size = struct.unpack('!Q', length_data)[0]
            
            # Stream from socket directly to temp file
            bytes_received = 0
            with open(temp_filename, 'wb') as f:
                while bytes_received < expected_size:
                    remaining = expected_size - bytes_received
                    chunk = sock.recv(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("Connection broke while receiving weight payload.")
                    f.write(chunk)
                    bytes_received += len(chunk)

            # Load from the completed temporary file
            transfer_data = torch.load(temp_filename, map_location='cpu', weights_only=False)

            transfer_time = (time.time() - start_time) * 1000
            throughput = expected_size / 1e6 / (transfer_time / 1000) if transfer_time > 0 else 0
            
            source_model_state = transfer_data.get('model_state_dict')
            source_optimizer_state = transfer_data.get('optimizer_state_dict')
            source_ema_loss = transfer_data.get('ema_loss', float('inf'))
            
            self.logger.log_event("WEIGHT_RECEIVE_COMPLETE",
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 data_size_bytes=expected_size,
                                 transfer_time_ms=transfer_time,
                                 fitness=source_ema_loss,
                                 message=f"Throughput: {throughput:.2f} MB/s")
            
            return source_model_state, source_optimizer_state, source_ema_loss, source_ema_loss
            
        except Exception as e:
            self.logger.log_event("WEIGHT_RECEIVE_ERROR", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 message=str(e))
            return None, None, None, None
        finally:
            # Robustly clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError as e:
                    self.logger.log_event("TEMP_FILE_CLEANUP_ERROR", message=f"Failed to remove {temp_filename}: {e}")
    
    def stop_gossip_protocol(self):
        """Stop the gossip protocol"""
        self.gossip_running = False
        if self.gossip_thread:
            self.gossip_thread.join(timeout=5)
        if self.server_thread:
            self.server_thread.join(timeout=5)
        self.logger.info("Gossip protocol stopped.")
    
    def request_mix(self):
        """Called by main thread to trigger mixing (for compatibility)"""
        # In the threaded approach, mixing happens automatically
        # This method exists for compatibility with existing code
        pass
    
    def get_status(self) -> dict:
        """Returns a comprehensive dictionary of the node's current status."""
        outbound = self.outbound_mixes_attempted
        inbound = self.inbound_mixes_attempted
        won = self.mixes_won
        lost = self.mixes_lost
        
        # A mix fails if it was attempted (inbound or outbound) but did not result in a win or loss.
        # This can happen due to timeouts, network errors, etc.
        failed_mixes = (outbound + inbound) - (won + lost)

        status = {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'peer_count': len(self.peer_list),
            'successful_mixes': self.successful_mixes,
            'current_step': self.current_step,
            
            # --- NEW METRICS FOR LOGGING ---
            'initiated_mixes': outbound,
            'received_mixes': inbound,
            'won_mixes': won,
            'lost_mixes': lost,
            'failed_mixes': failed_mixes
        }
        # Conditionally add the lock metric
        if self.use_node_local_lock:
            status['skipped_due_to_lock'] = self.mixes_skipped_locked
        
        return status

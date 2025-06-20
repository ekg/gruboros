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
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils
from .structured_logger import GossipLogger

@dataclass
class WeightUpdate:
    state_dict: dict
    source_node: str
    source_fitness: float  # Added fitness transfer
    source_ema_loss: Optional[float]  # Added EMA loss transfer
    correlation_id: str    # Added correlation tracking

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module,
                 global_rank: int, local_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01, 
                 output_dir: Optional[str] = None):
        # Store parameters as instance variables FIRST
        self.node_id = node_id
        self.model = model  # Main thread owns this
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        self.mixing_probability = mixing_probability
        
        # Now initialize other attributes
        self.mixing_rng = random.Random(42 + self.global_rank * 1000)
        
        # Thread-safe communication
        self.incoming_updates = queue.Queue()  # Background thread -> Main thread
        self.step_notifications = queue.Queue()  # Main thread -> Background thread
        self.fitness_tracker = FitnessTracker()
        self.fitness_lock = threading.Lock()
        
        # NEW: Add async weight management
        self.pending_update = None
        self.weight_stream = torch.cuda.Stream()  # Dedicated stream for weight ops
        self.weight_ready_event = torch.cuda.Event()  # Synchronization event
        self.weight_transfer_lock = threading.Lock()
        
        # Replace standard logger with structured gossip logger
        self.logger = GossipLogger(self.node_id, self.global_rank, self.local_rank, self.data_parallel_rank, output_dir)
        
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
    
    def apply_pending_update(self) -> bool:
        """Apply any pending weight update (call at start of training step)"""
        if self.pending_update is None:
            return False
        
        with self.weight_transfer_lock:
            # Wait for async transfer to complete
            torch.cuda.current_stream().wait_event(self.weight_ready_event)
            
            # Now apply the update (this is the only blocking part, but it's fast)
            start_time = time.time()
            self.model.load_state_dict(self.pending_update.state_dict)
            load_time = (time.time() - start_time) * 1000
            
            # Inherit fitness
            with self.fitness_lock:
                source_ema_loss = getattr(self.pending_update, 'source_ema_loss', None)
                self.fitness_tracker.inherit_fitness(
                    self.pending_update.source_fitness, 
                    source_ema_loss
                )
            
            self.successful_mixes += 1
            
            # Removed verbose logging to prevent stalls
            
            self.pending_update = None
            return True
    
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
        
        # Discover peers
        self.peer_list = {}
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
                        self.logger.log_event("MIX_DECISION", 
                                             step=step,
                                             fitness=self.get_current_fitness(),
                                             message="Random check triggered mixing attempt")
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
        
        try:
            NetworkUtils.optimize_socket(client_sock)
            client_sock.settimeout(5.0)  # Short timeout for initial fitness exchange
            
            # Receive fitness comparison with correlation ID
            data = client_sock.recv(1024).decode()
            if not data.startswith("FITNESS:"):
                return
            
            # Parse: FITNESS:0.123456:CID:abc12345
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
            
            if our_fitness > peer_fitness:
                # We win - send our weights
                client_sock.send(b"SENDING_WEIGHTS")
                # Extend timeout for weight transfer
                client_sock.settimeout(120.0)
                self._send_our_weights_to_peer(client_sock, correlation_id)
                
            else:
                # We lose - receive their weights
                client_sock.send(b"SEND_ME_WEIGHTS")
                # Extend timeout for weight transfer
                client_sock.settimeout(120.0)
                new_weights, source_fitness, source_ema_loss = self._receive_weights_from_peer(client_sock, correlation_id)
                
                if new_weights:
                    # Queue the update for main thread (don't apply here!)
                    update = WeightUpdate(
                        state_dict=new_weights,
                        source_node=peer_addr,
                        source_fitness=source_fitness,
                        source_ema_loss=source_ema_loss,
                        correlation_id=correlation_id
                    )
                    self.incoming_updates.put(update)
                    self.logger.log_event("WEIGHT_UPDATE_QUEUED", 
                                         step=self.current_step,
                                         correlation_id=correlation_id,
                                         peer_addr=peer_addr,
                                         fitness=source_fitness)
                
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
            message = f"FITNESS:{our_fitness:.6f}:CID:{correlation_id}".encode()
            sock.send(message)
            
            # Keep short timeout for fitness comparison response
            sock.settimeout(2.0)
            response = sock.recv(1024)
            
            # Extend timeout for weight transfers
            sock.settimeout(120.0)
            
            if response == b"SENDING_WEIGHTS":
                new_weights, source_fitness, source_ema_loss = self._receive_weights_from_peer(sock, correlation_id)
                if new_weights:
                    update = WeightUpdate(
                        state_dict=new_weights,
                        source_node=peer_address,
                        source_fitness=source_fitness,
                        source_ema_loss=source_ema_loss,
                        correlation_id=correlation_id
                    )
                    self.incoming_updates.put(update)
                    
            elif response == b"SEND_ME_WEIGHTS":
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
        """Send weights with performance monitoring"""
        start_time = time.time()
        
        try:
            state_dict = self.model.state_dict()
            cpu_state_dict = {name: param.cpu() for name, param in state_dict.items()}
            
            # Include fitness and raw EMA loss in the transfer
            our_fitness = self.get_current_fitness()
            our_ema_loss = self.fitness_tracker.get_recent_loss()
            transfer_data = {
                'state_dict': cpu_state_dict,
                'fitness': our_fitness,
                'ema_loss': our_ema_loss,
                'correlation_id': correlation_id
            }
            
            buffer = io.BytesIO()
            torch.save(transfer_data, buffer)
            weights_bytes = buffer.getvalue()
            
            # Send length prefix then data
            sock.send(struct.pack('!I', len(weights_bytes)))
            
            chunk_size = 1024 * 1024  # 1MB chunks
            bytes_sent = 0
            
            while bytes_sent < len(weights_bytes):
                chunk_end = min(bytes_sent + chunk_size, len(weights_bytes))
                chunk = weights_bytes[bytes_sent:chunk_end]
                sock.sendall(chunk)
                bytes_sent = chunk_end
            
            transfer_time = (time.time() - start_time) * 1000
            throughput = len(weights_bytes) / 1e6 / (transfer_time / 1000)
            
            self.logger.log_event("WEIGHT_SEND_COMPLETE",
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 data_size_bytes=len(weights_bytes),
                                 transfer_time_ms=transfer_time,
                                 fitness=our_fitness,
                                 message=f"Throughput: {throughput:.2f} MB/s")
            
        except Exception as e:
            self.logger.log_event("WEIGHT_SEND_ERROR", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 message=str(e))
    
    def _receive_weights_from_peer(self, sock, correlation_id: str) -> tuple[Optional[dict], Optional[float], Optional[float]]:
        """Receive weights with performance monitoring"""
        start_time = time.time()
        
        try:
            # Receive length prefix
            length_data = b''
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None, None
                length_data += chunk
            
            expected_size = struct.unpack('!I', length_data)[0]
            
            # Receive data
            received_data = bytearray()
            while len(received_data) < expected_size:
                remaining = expected_size - len(received_data)
                chunk_size = min(1024 * 1024, remaining)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    return None, None
                received_data.extend(chunk)
            
            # Deserialize
            buffer = io.BytesIO(bytes(received_data))
            transfer_data = torch.load(buffer, map_location='cpu', weights_only=False)
            
            transfer_time = (time.time() - start_time) * 1000
            throughput = expected_size / 1e6 / (transfer_time / 1000)
            
            source_fitness = transfer_data.get('fitness', 0.0)
            source_ema_loss = transfer_data.get('ema_loss', None)
            
            self.logger.log_event("WEIGHT_RECEIVE_COMPLETE",
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 data_size_bytes=expected_size,
                                 transfer_time_ms=transfer_time,
                                 fitness=source_fitness,
                                 message=f"Throughput: {throughput:.2f} MB/s")
            
            return transfer_data['state_dict'], source_fitness, source_ema_loss
            
        except Exception as e:
            self.logger.log_event("WEIGHT_RECEIVE_ERROR", 
                                 step=self.current_step,
                                 correlation_id=correlation_id,
                                 message=str(e))
            return None, None, None
    
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
        return {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'recent_loss': self.fitness_tracker.get_recent_loss(),
            'peer_count': len(self.peer_list),
            'mixing_attempts': self.mixing_attempts,
            'successful_mixes': self.successful_mixes,
            'current_step': self.current_step
        }

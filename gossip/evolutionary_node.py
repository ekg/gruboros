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

@dataclass
class WeightUpdate:
    state_dict: dict
    source_node: str

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module,
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01):
        self.node_id = node_id
        self.model = model  # Main thread owns this
        self.global_rank = global_rank
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_probability = mixing_probability
        
        # Thread-safe communication
        self.incoming_updates = queue.Queue()  # Background thread -> Main thread
        self.step_notifications = queue.Queue()  # Main thread -> Background thread
        self.fitness_tracker = FitnessTracker()
        self.fitness_lock = threading.Lock()
        
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.gossip_port = NetworkUtils.get_gossip_port(data_parallel_rank, master_addr)
        
        # Thread control
        self.gossip_running = False
        self.gossip_thread = None
        self.server_thread = None
        
        self.logger = logging.getLogger(f'evolutionary_node_{self.node_id}')
        
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(
            global_rank, world_size, data_parallel_rank, tp_size, self.logger
        )
        
        self.mixing_attempts = 0
        self.successful_mixes = 0
        self.current_step = 0
        
    def update_fitness(self, loss_value: float, step: int):
        """Called by main thread after each training step"""
        with self.fitness_lock:
            self.fitness_tracker.update(loss_value)
            
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
        """Called by main thread to check for new weights"""
        try:
            # Non-blocking check - returns immediately
            update = self.incoming_updates.get_nowait()
            return update
        except queue.Empty:
            return None
    
    def apply_update(self, update: WeightUpdate):
        """Called by main thread to safely apply update"""
        self.logger.info(f"🔄 Applying update from {update.source_node}")
        device = next(self.model.parameters()).device
        
        # Move state dict to correct device
        for name, param in update.state_dict.items():
            update.state_dict[name] = param.to(device)
            
        self.model.load_state_dict(update.state_dict)
        self.successful_mixes += 1
        self.logger.info("✅ Update applied successfully")

    def start_gossip_protocol(self):
        """Start the background gossip thread"""
        self.gossip_running = True
        self.gossip_thread = threading.Thread(target=self._gossip_worker, daemon=True)
        self.gossip_thread.start()
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        self.logger.info("🚀 Background gossip started")
    
    def _gossip_worker(self):
        """Background thread that handles gossip networking - NOW STEP-BASED"""
        # Wait a bit for server to start
        time.sleep(2)
        
        # Discover peers
        self.peer_list = {}
        for bootstrap_addr in self.bootstrap_nodes:
            self.peer_list[bootstrap_addr] = {}
        
        self.logger.info(f"Discovered {len(self.peer_list)} peers: {list(self.peer_list.keys())}")
        
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
                        self.logger.info(f"🎲 Step {step}: Attempting evolutionary mix")
                        self._try_mix_with_peer()
                    
                    # Mark step notification as processed
                    self.step_notifications.task_done()
                    
                except queue.Empty:
                    # Timeout - just continue the loop to check gossip_running
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in gossip worker: {e}")
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
            self.logger.info(f"🚀 Gossip server listening on port {self.gossip_port}")
            
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
                        self.logger.error(f"Server error: {e}")
                        time.sleep(1)
                    
        finally:
            server_sock.close()
    
    def _handle_incoming_request(self, client_sock, addr):
        """Background thread: handle one incoming gossip request"""
        try:
            NetworkUtils.optimize_socket(client_sock)
            
            # Receive fitness comparison
            data = client_sock.recv(1024).decode()
            if not data.startswith("FITNESS:"):
                return
                
            peer_fitness = float(data.split(":")[1])
            our_fitness = self.get_current_fitness()
            
            self.logger.info(f"📞 Step {self.current_step}: Peer {addr[0]} fitness: {peer_fitness:.4f}, ours: {our_fitness:.4f}")
            
            if our_fitness > peer_fitness:
                # We win - send our weights
                client_sock.send(b"SENDING_WEIGHTS")
                self._send_our_weights_to_peer(client_sock)
                self.logger.info(f"🏆 Step {self.current_step}: Sent weights to weaker peer")
                
            else:
                # We lose - receive their weights
                client_sock.send(b"SEND_ME_WEIGHTS")
                new_weights = self._receive_weights_from_peer(client_sock)
                
                if new_weights:
                    # Queue the update for main thread (don't apply here!)
                    update = WeightUpdate(
                        state_dict=new_weights,
                        source_node=f"peer_{addr[0]}"
                    )
                    self.incoming_updates.put(update)
                    self.logger.info(f"📦 Step {self.current_step}: Queued weight update for main thread")
                
        except Exception as e:
            self.logger.error(f"Error handling request from {addr}: {e}")
        finally:
            client_sock.close()
    
    def _try_mix_with_peer(self):
        """Background thread: try to connect to a peer and mix"""
        if not self.peer_list:
            return
            
        # Choose random peer
        my_address = f"{os.environ.get('MASTER_ADDR', 'localhost')}:{self.gossip_port}"
        other_peers = [p for p in self.peer_list.keys() if p != my_address]
        if not other_peers:
            return
            
        peer_address = self.mixing_rng.choice(other_peers)
        host, port_str = peer_address.split(':')
        port = int(port_str)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        NetworkUtils.optimize_socket(sock)
        
        try:
            sock.settimeout(10.0)
            sock.connect((host, port))
            
            # Send our fitness
            our_fitness = self.get_current_fitness()
            message = f"FITNESS:{our_fitness:.6f}".encode()
            sock.send(message)
            
            # Get response
            response = sock.recv(1024)
            
            if response == b"SENDING_WEIGHTS":
                # Peer is sending weights to us
                new_weights = self._receive_weights_from_peer(sock)
                if new_weights:
                    update = WeightUpdate(
                        state_dict=new_weights,
                        source_node=peer_address
                    )
                    self.incoming_updates.put(update)
                    self.logger.info(f"📦 Step {self.current_step}: Queued weight update from {peer_address}")
                    
            elif response == b"SEND_ME_WEIGHTS":
                # We need to send weights to peer
                self._send_our_weights_to_peer(sock)
                self.logger.info(f"🏆 Step {self.current_step}: Sent weights to weaker peer {peer_address}")
                
            self.mixing_attempts += 1
            
        except Exception as e:
            self.logger.warning(f"❌ Step {self.current_step}: Could not connect to {peer_address}: {e}")
        finally:
            sock.close()
    
    def _send_our_weights_to_peer(self, sock):
        """Send our current weights through socket"""
        try:
            # Serialize weights
            state_dict = self.model.state_dict()
            cpu_state_dict = {name: param.cpu() for name, param in state_dict.items()}
            
            buffer = io.BytesIO()
            torch.save(cpu_state_dict, buffer)
            weights_bytes = buffer.getvalue()
            
            self.logger.info(f"🚀 Sending {len(weights_bytes)/1e6:.2f} MB to peer")
            
            # Send length prefix
            sock.send(struct.pack('!I', len(weights_bytes)))
            
            # Send data in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            bytes_sent = 0
            
            while bytes_sent < len(weights_bytes):
                chunk_end = min(bytes_sent + chunk_size, len(weights_bytes))
                chunk = weights_bytes[bytes_sent:chunk_end]
                sock.sendall(chunk)
                bytes_sent = chunk_end
            
            self.logger.info("✅ Weight transfer completed")
            
        except Exception as e:
            self.logger.error(f"Error sending weights: {e}")
    
    def _receive_weights_from_peer(self, sock) -> Optional[dict]:
        """Receive weights from peer through socket"""
        try:
            # Receive length prefix
            length_data = b''
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk
            
            expected_size = struct.unpack('!I', length_data)[0]
            self.logger.info(f"🔄 Receiving {expected_size/1e6:.2f} MB from peer")
            
            # Receive data
            received_data = bytearray()
            while len(received_data) < expected_size:
                remaining = expected_size - len(received_data)
                chunk_size = min(1024 * 1024, remaining)  # 1MB chunks
                chunk = sock.recv(chunk_size)
                if not chunk:
                    return None
                received_data.extend(chunk)
            
            # Deserialize
            buffer = io.BytesIO(bytes(received_data))
            state_dict = torch.load(buffer, map_location='cpu')
            
            self.logger.info("✅ Weights received and deserialized")
            return state_dict
            
        except Exception as e:
            self.logger.error(f"Error receiving weights: {e}")
            return None
    
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

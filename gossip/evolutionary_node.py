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
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils
from .structured_logger import GossipLogger
from .filesystem_coordinator import FilesystemCoordinator
import tempfile
import uuid
import fcntl
import contextlib
import gc
import psutil

def log_memory_usage(rank: int, message: str):
    """Logs current memory usage for debugging"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    print(f"[Rank {rank}] {message}: Process RSS: {rss_gb:.2f} GB, Node Available: {available_gb:.2f} GB", flush=True)

def check_memory_headroom(rank: int, required_bytes: int, message: str) -> bool:
    """Check if there's enough memory headroom for an operation"""
    safety_buffer_bytes = 4 * 1024**3  # 4 GB safety buffer
    total_required = required_bytes + safety_buffer_bytes
    
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    required_gb = total_required / (1024**3)
    
    if available_gb < required_gb:
        print(f"[Rank {rank}] MEMORY WARNING - {message}: Need {required_gb:.2f} GB, only {available_gb:.2f} GB available", flush=True)
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
                 save_callback: Optional[callable] = None):
        
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
        
        self.coordinator: Optional[FilesystemCoordinator] = None
        if self.use_filesystem_coordinator:
            self.coordinator = FilesystemCoordinator(
                global_rank=self.global_rank, world_size=self.world_size, output_dir=output_dir
            )
            self.coordinator.set_fitness_provider(self.get_current_fitness)
        
        self.mixing_rng = random.Random(42 + self.global_rank * 1000)
        self.incoming_updates = queue.Queue()
        self.step_notifications = queue.Queue()
        self.fitness_tracker = FitnessTracker(window_size=fitness_window_size)
        self.fitness_lock = threading.Lock()
        
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
        with self.fitness_lock: self.fitness_tracker.update(loss_value)
        try: self.step_notifications.put_nowait(step)
        except queue.Full: pass

    def get_current_fitness(self) -> float:
        with self.fitness_lock: return self.fitness_tracker.get_fitness()

    def check_for_updates(self) -> Optional[WeightUpdate]:
        try:
            update = self.incoming_updates.get_nowait()
            self._start_async_weight_transfer(update)
            return update
        except queue.Empty: return None

    def _start_async_weight_transfer(self, update: WeightUpdate):
        if self.use_node_local_lock:
            try:
                with file_lock(self.node_lock_path, timeout=15.0):
                    self._load_and_transfer_payload(update)
            except TimeoutError:
                self.logger.log_event("UPDATE_SKIPPED_LOCK", message="Node lock busy, skipping weight update application.")
                if os.path.exists(update.payload_path): os.remove(update.payload_path)
        else:
            self._load_and_transfer_payload(update)

    def _load_and_transfer_payload(self, update: WeightUpdate):
        with self.weight_transfer_lock:
            try:
                payload_size = os.path.getsize(update.payload_path)
                log_memory_usage(self.global_rank, "Before torch.load")
                
                if not check_memory_headroom(self.global_rank, payload_size, "torch.load operation"):
                    self.logger.log_event("UPDATE_SKIPPED_MEMORY", message="Insufficient memory headroom, skipping update")
                    if os.path.exists(update.payload_path): os.remove(update.payload_path)
                    return

                loaded_data = torch.load(update.payload_path, map_location='cpu')
                log_memory_usage(self.global_rank, "After torch.load")
                
                setattr(update, 'state_dict', loaded_data['model_state_dict'])
                setattr(update, 'optimizer_state_dict', loaded_data.get('optimizer_state_dict'))
                
                os.remove(update.payload_path)
                
                with torch.cuda.stream(self.weight_stream):
                    device = next(self.model.parameters()).device
                    for name, param in update.state_dict.items():
                        update.state_dict[name] = param.pin_memory().to(device, non_blocking=True)
                    self.weight_ready_event.record(self.weight_stream)
                
                self.pending_update = update
                
                # Critical memory release sequence
                del loaded_data['model_state_dict']
                if 'optimizer_state_dict' in loaded_data:
                    del loaded_data['optimizer_state_dict']
                del loaded_data
                gc.collect()
                time.sleep(0.1)  # Brief pause for OS to reclaim memory
                
                log_memory_usage(self.global_rank, "After memory cleanup and before lock release")

            except Exception as e:
                self.logger.log_event("PAYLOAD_LOAD_ERROR", message=str(e))
                if os.path.exists(update.payload_path): os.remove(update.payload_path)
                self.pending_update = None

    def apply_pending_update(self) -> tuple[bool, bool]:
        if self.pending_update is None: return False, False
        with self.weight_transfer_lock:
            torch.cuda.current_stream().wait_event(self.weight_ready_event)
            needs_optimizer_reset = False
            
            if self.merge_method == 'recombination':
                winner_model_state = self.pending_update.state_dict
                alpha = self.recombination_alpha
                with torch.no_grad():
                    for name, loser_param in self.model.named_parameters():
                        if name in winner_model_state and loser_param.dtype.is_floating_point:
                            winner_param = winner_model_state[name].to(loser_param.device)
                            loser_param.mul_(1.0 - alpha).add_(winner_param, alpha=alpha)
                
                if self.optimizer_recombination == 'interpolate' and self.pending_update.optimizer_state_dict:
                    winner_optim_state = self.pending_update.optimizer_state_dict['state']
                    loser_optim_state = self.optimizer.state
                    with torch.no_grad():
                        for p_id, winner_p_state in winner_optim_state.items():
                            if p_id in loser_optim_state:
                                loser_p_state = loser_optim_state[p_id]
                                for key in ['exp_avg', 'exp_avg_sq']:
                                    if key in loser_p_state and key in winner_p_state:
                                        loser_tensor = loser_p_state[key]
                                        winner_tensor = winner_p_state[key].to(loser_tensor.device)
                                        loser_tensor.mul_(1.0 - alpha).add_(winner_tensor, alpha=alpha)
                elif self.optimizer_recombination == 'reset': needs_optimizer_reset = True
            else:
                self.model.load_state_dict(self.pending_update.state_dict)
                if self.pending_update.optimizer_state_dict: 
                    self.optimizer.load_state_dict(self.pending_update.optimizer_state_dict)
                with self.fitness_lock: self.fitness_tracker.inherit_fitness(self.pending_update.source_ema_loss)
            
            self.successful_mixes += 1; self.pending_update = None
            return True, needs_optimizer_reset

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
        peer_addr = f"{addr[0]}:{addr[1]}"; correlation_id = None; self.inbound_mixes_attempted += 1
        try:
            NetworkUtils.optimize_socket(client_sock); client_sock.settimeout(5.0)
            data = client_sock.recv(1024).decode()
            if not data.startswith("EMA_LOSS:"): return
            parts = data.split(":"); peer_fitness, correlation_id = (float(parts[1]), parts[3]) if len(parts) >= 4 and parts[2] == "CID" else (float(parts[1]), self.logger.generate_correlation_id())
            our_fitness = self.get_current_fitness()
            self.logger.log_event("INCOMING_FITNESS_COMPARISON", step=self.current_step, fitness=our_fitness, correlation_id=correlation_id, peer_addr=peer_addr, message=f"peer_fitness={peer_fitness:.4f}")
            
            if our_fitness < peer_fitness:
                self.mixes_won += 1
                # Tell peer if we'll send optimizer state
                send_optimizer = self.optimizer_recombination == 'interpolate'
                client_sock.send(f"SENDING_WEIGHTS:{send_optimizer}".encode())
                client_sock.settimeout(120.0)
                self._send_our_weights_to_peer(client_sock, correlation_id, send_optimizer)
                if self.save_callback:
                    try: self.save_callback(self.current_step, our_fitness, opportunistic=True)
                    except Exception as e: self.logger.log_event("OPPORTUNISTIC_SAVE_FAILED", message=str(e))
            else:
                self.mixes_lost += 1
                # Tell peer our optimizer policy so they know what to send
                client_sock.send(f"SEND_ME_WEIGHTS:{self.optimizer_recombination}".encode())
                client_sock.settimeout(120.0)
                payload_path, source_ema_loss = self._receive_weights_from_peer(client_sock, correlation_id)
                if payload_path:
                    update = WeightUpdate(payload_path=payload_path, source_node=peer_addr, source_ema_loss=source_ema_loss, correlation_id=correlation_id)
                    self.incoming_updates.put(update); self.logger.log_event("WEIGHT_UPDATE_QUEUED", step=self.current_step, correlation_id=correlation_id, peer_addr=peer_addr, fitness=source_ema_loss)
        except Exception as e: self.logger.log_event("INCOMING_REQUEST_ERROR", step=self.current_step, correlation_id=correlation_id, peer_addr=peer_addr, message=str(e))
        finally: client_sock.close()

    def _try_mix_with_peer(self):
        if not self.peer_list: return
        self.outbound_mixes_attempted += 1; correlation_id = self.logger.generate_correlation_id()
        local_identity = self.logger.node_identity; other_peers = [p for p in self.peer_list.keys() if p != local_identity]
        if not other_peers: return
        peer_address = self.mixing_rng.choice(other_peers)
        self.logger.log_event("MIX_ATTEMPT_START", step=self.current_step, correlation_id=correlation_id, peer_addr=peer_address, fitness=self.get_current_fitness())
        host, port = peer_address.split(':'); sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); NetworkUtils.optimize_socket(sock)
        try:
            sock.settimeout(3.0); sock.connect((host, int(port)))
            our_fitness = self.get_current_fitness(); sock.send(f"EMA_LOSS:{our_fitness:.6f}:CID:{correlation_id}".encode()); sock.settimeout(5.0); response_str = sock.recv(1024).decode()
            sock.settimeout(120.0)
            
            if response_str.startswith("SENDING_WEIGHTS"):
                self.mixes_lost += 1
                payload_path, source_ema_loss = self._receive_weights_from_peer(sock, correlation_id)
                if payload_path: self.incoming_updates.put(WeightUpdate(payload_path=payload_path, source_node=peer_address, source_ema_loss=source_ema_loss, correlation_id=correlation_id))
            elif response_str.startswith("SEND_ME_WEIGHTS"):
                self.mixes_won += 1
                # Check if peer needs optimizer state
                loser_optimizer_policy = response_str.split(':')[-1] if ':' in response_str else 'reset'
                send_optimizer = loser_optimizer_policy == 'interpolate'
                self._send_our_weights_to_peer(sock, correlation_id, send_optimizer)
                if self.save_callback:
                    try: self.save_callback(self.current_step, our_fitness, opportunistic=True)
                    except Exception as e: self.logger.log_event("OPPORTUNISTIC_SAVE_FAILED", message=str(e))
            self.mixing_attempts += 1
        except Exception as e: self.logger.log_event("MIX_ATTEMPT_FAILED", step=self.current_step, correlation_id=correlation_id, peer_addr=peer_address, message=str(e))
        finally: sock.close()

    def _send_our_weights_to_peer(self, sock, correlation_id: str, send_optimizer_state: bool = True):
        start_time = time.time(); temp_filename = os.path.join(self.gossip_temp_dir, f"gossip_payload_{uuid.uuid4()}.pt")
        try:
            log_memory_usage(self.global_rank, "Before creating payload")
            
            model_cpu_state = {name: param.cpu() for name, param in self.model.state_dict().items()}
            our_ema_loss = self.get_current_fitness()
            transfer_data = {'model_state_dict': model_cpu_state, 'ema_loss': our_ema_loss}
            
            # Only include optimizer state if requested by the receiving peer
            if send_optimizer_state:
                opt_state_dict = self.optimizer.state_dict(); optimizer_cpu_state = {'param_groups': opt_state_dict['param_groups'], 'state': {}}
                for param_id, param_state in opt_state_dict['state'].items():
                    optimizer_cpu_state['state'][param_id] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in param_state.items()}
                transfer_data['optimizer_state_dict'] = optimizer_cpu_state
                log_memory_usage(self.global_rank, "After creating optimizer payload")
            else:
                log_memory_usage(self.global_rank, "Skipping optimizer state (not requested)")
            
            torch.save(transfer_data, temp_filename)
            file_size = os.path.getsize(temp_filename); sock.send(struct.pack('!Q', file_size))
            with open(temp_filename, 'rb') as f:
                while chunk := f.read(4 * 1024 * 1024): sock.sendall(chunk)
            self.logger.log_event("WEIGHT_SEND_COMPLETE", step=self.current_step, correlation_id=correlation_id, data_size_bytes=file_size, transfer_time_ms=(time.time() - start_time) * 1000, fitness=our_ema_loss, message=f"Optimizer included: {send_optimizer_state}")
        except Exception as e: self.logger.log_event("WEIGHT_SEND_ERROR", step=self.current_step, correlation_id=correlation_id, message=str(e))
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
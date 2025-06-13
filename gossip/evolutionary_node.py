import asyncio
import random
import numpy as np
import torch
import json
import time
import os
import socket
import pickle
import hashlib
import logging
from asyncio import Queue
from typing import Dict, List, Optional, Set
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module, 
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        
        # Per-process random state
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_probability = mixing_probability
        self.is_mixing = False  # State flag to prevent concurrent mixes
        
        # Queue for mix requests from training loop
        self.mix_request_queue = Queue(maxsize=1)
        
        # Global lock to prevent concurrent mixing operations
        self.mixing_lock = asyncio.Lock()
        
        # Fitness tracking
        self.fitness_tracker = FitnessTracker()
        self.step_count = 0
        
        # Peer management
        self.peer_list: Dict[str, dict] = {}
        
        # NEW: Scheduled mixing system
        self.pending_mixes: Dict[int, dict] = {}  # {step: {partner, role}}
        self.mixing_proposals: Dict[str, dict] = {}  # {partner: proposal_data}
        
        # Network setup
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.gossip_port = NetworkUtils.get_gossip_port(data_parallel_rank, master_addr)
        self.server = None
        self.gossip_running = False
        self.gossip_task = None
        self.discovery_task = None
        
        # Logging
        self.logger = logging.getLogger(f'evolutionary_node_{node_id}')
        
        # Bootstrap nodes - NOW INCLUDES TP_SIZE
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(
            global_rank, world_size, data_parallel_rank, tp_size, self.logger
        )
        
        # Statistics
        self.mixing_attempts = 0
        self.successful_mixes = 0
        
    def update_fitness(self, loss_value: float):
        """Update fitness based on recent loss"""
        self.fitness_tracker.update(loss_value)
        self.step_count += 1
    
    def get_current_fitness(self) -> float:
        """Get current fitness score"""
        return self.fitness_tracker.get_fitness()
    
    # The attempt_mix_if_scheduled method has been removed.
    # Mixing is now handled by the persistent _mixer_task background thread.
    
    async def _execute_mixing_as_initiator(self, partner: str) -> bool:
        """Execute mixing as the initiator"""
        try:
            host, port = partner.split(':')
            port = int(port)
            
            # Quick connection for weight exchange
            connection = await NetworkUtils.safe_connect(host, port, timeout=3.0)
            if not connection:
                return False
            
            reader, writer = connection
            
            try:
                # Send "ready to mix" signal
                ready_msg = {
                    'type': 'ready_to_mix',
                    'sender': self.node_id,
                    'fitness': self.get_current_fitness()
                }
                
                if not await NetworkUtils.safe_send_json(writer, ready_msg, timeout=2.0):
                    return False
                
                # Get partner's weights
                partner_weights = await NetworkUtils.safe_recv_json(reader, timeout=5.0)
                if not partner_weights or partner_weights.get('type') != 'weights':
                    return False
                
                # Send our weights (full state dict)
                our_weights = {
                    'type': 'weights',
                    'fitness': self.get_current_fitness(),
                    'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
                    'weights_hash': self._get_model_hash()
                }
                
                if not await NetworkUtils.safe_send_json(writer, our_weights, timeout=5.0):
                    return False
                
                # Convert partner's weights back to tensors and do the cloning
                if 'state_dict' in partner_weights:
                    partner_state = {k: torch.tensor(v) for k, v in partner_weights['state_dict'].items()}
                    self._mix_weights_based_on_fitness(partner_weights['fitness'], partner_state)
                else:
                    self._mix_weights_based_on_fitness(partner_weights['fitness'])
                
                return True
                
            finally:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            self.logger.error(f"Mixing execution failed: {e}")
            return False
    
    async def _execute_mixing_as_acceptor(self, partner: str) -> bool:
        """Execute mixing as the acceptor - wait for initiator to connect"""
        # The acceptor waits for the initiator to connect
        # This is handled in _handle_ready_to_mix
        return True
    
    def _mix_weights_based_on_fitness(self, partner_fitness: float, partner_weights: dict = None):
        """Perform pure evolutionary cloning - loser is completely overwritten by winner"""
        current_fitness = self.get_current_fitness()
        recent_loss = self.fitness_tracker.get_recent_loss()
        
        # Determine if partner is better (higher fitness = lower loss = better model)
        partner_is_better = partner_fitness > current_fitness
        
        self.logger.info(f"🧬 EVOLUTIONARY CLONING:")
        self.logger.info(f"  Partner fitness: {partner_fitness:.4f}, Our fitness: {current_fitness:.4f}")
        self.logger.info(f"  Our recent loss: {recent_loss:.4f}")
        
        if partner_is_better:
            # We are the losing model - complete cloning of partner's weights
            self.logger.info(f"🔥 LOSING MODEL: Complete takeover by superior partner")
            
            if partner_weights is not None:
                # Load partner's weights directly
                try:
                    self.model.load_state_dict(partner_weights)
                    total_params = sum(p.numel() for p in self.model.parameters())
                    self.logger.info(f"  🚨 COMPLETE CLONING: All {total_params:,} parameters replaced with winner's weights")
                except Exception as e:
                    self.logger.error(f"Failed to clone partner weights: {e}")
            else:
                self.logger.warning("No partner weights provided for cloning")
        else:
            # We are the winning model - no changes needed
            self.logger.info(f"🏆 WINNING MODEL: No changes - maintaining superior weights")
            self.logger.info(f"  ✨ WINNER PRESERVATION: Keeping all parameters unchanged")
    
    def _get_model_hash(self) -> str:
        """Get hash of model weights"""
        model_str = ""
        for name, param in self.model.named_parameters():
            model_str += param.data.cpu().numpy().tobytes().hex()[:50]
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    
    
    async def _handle_peer_connection(self, reader, writer):
        """Handle incoming connections with global lock to prevent concurrent mixing"""
        # Use the global lock to ensure we don't handle a request while
        # we are busy initiating our own mix.
        async with self.mixing_lock:
            try:
                await self._run_gossip_protocol(reader, writer, is_initiator=False)
            except Exception as e:
                self.logger.error(f"Error in _handle_peer_connection: {e}")
            # The connection is managed by the initiator - we don't close it here
    
    async def _handle_mixing_proposal(self, reader, writer, proposal):
        """Handle mixing proposal - schedule future mixing"""
        partner_id = proposal.get('sender')
        partner_fitness = proposal.get('fitness', 0.0)
        proposed_step = proposal.get('mix_at_step')
        
        current_fitness = self.get_current_fitness()
        recent_loss = self.fitness_tracker.get_recent_loss()
        
        # Always accept proposals (we're async now!)
        accept = proposed_step not in self.pending_mixes
        
        self.logger.info(f"📨 RECEIVED MIXING PROPOSAL: from={partner_id}, step={proposed_step}")
        self.logger.info(f"  Partner fitness: {partner_fitness:.4f} vs our fitness: {current_fitness:.4f}")
        self.logger.info(f"  Our recent loss: {recent_loss:.4f}, fitness ratio: {partner_fitness/max(current_fitness, 1e-6):.3f}")
        
        if accept:
            # Schedule the mixing
            self.pending_mixes[proposed_step] = {
                'partner': partner_id,
                'role': 'acceptor',
                'fitness': partner_fitness
            }
            self.logger.info(f"✅ ACCEPTED PROPOSAL: scheduling mix at step {proposed_step}")
        else:
            self.logger.info(f"❌ REJECTED PROPOSAL: step {proposed_step} already busy")
        
        response = {
            'accept': accept,
            'fitness': current_fitness
        }
        
        await NetworkUtils.safe_send_json(writer, response, timeout=2.0)
    
    async def _run_gossip_protocol(self, reader, writer, is_initiator: bool):
        """
        Executes the entire gossip protocol using proper length-prefixed messages
        to avoid mixing text and binary data.
        """
        partner_id = "unknown_peer"
        try:
            if is_initiator:
                # 1. INITIATOR: Send the probe using length-prefixed message
                partner_id = self.mixing_rng.choice(list(self.peer_list.keys()))
                current_fitness = self.get_current_fitness()
                self.logger.info(f"🎲 INITIATING MIXING with {partner_id} (fitness: {current_fitness:.4f})")
                probe = f"PROBE|{self.node_id}|{current_fitness:.6f}"
                await NetworkUtils.send_message(writer, probe.encode())

                # 2. INITIATOR: Wait for response
                response_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not response_data:
                    return

                response = response_data.decode()
                if response == "RESPONSE|LOSER":
                    # 3a. INITIATOR: Peer is loser, send weights
                    self.logger.info(f"🏆 Peer {partner_id} is loser. Sending weights.")
                    
                    # Get model weights as bytes
                    state_dict = self.model.state_dict()
                    weights_data = {}
                    for k, v in state_dict.items():
                        weights_data[k] = v.cpu().numpy().tolist()
                    
                    import pickle
                    weights_bytes = pickle.dumps(weights_data)
                    
                    # Send weights using length-prefixed message
                    await NetworkUtils.send_message(writer, weights_bytes)
                    
                    self.successful_mixes += 1
                    self.logger.info(f"✅ Sent weights to {partner_id}.")
                else:
                    self.logger.info(f"🥈 Peer {partner_id} is winner. Ending mix.")

            else:  # I am the PEER
                # 1. PEER: Receive the probe using length-prefixed message
                probe_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not probe_data:
                    return
                
                probe = probe_data.decode()
                if not probe.startswith("PROBE"):
                    self.logger.warning(f"Invalid probe received: {probe}")
                    return
                
                _, partner_id, peer_fitness_str = probe.split('|')
                peer_fitness = float(peer_fitness_str)
                current_fitness = self.get_current_fitness()
                self.logger.info(f"📬 Received probe from {partner_id} (fitness: {peer_fitness:.4f}). Our fitness: {current_fitness:.4f}")

                # 2. PEER: Decide and send response
                if peer_fitness > current_fitness:
                    # 3a. PEER: We are the loser, send LOSER response and wait for weights
                    self.logger.info(f"🥈 We are loser. Sending LOSER response to {partner_id}.")
                    await NetworkUtils.send_message(writer, b"RESPONSE|LOSER")

                    # 4. PEER: Receive weights using length-prefixed message
                    weights_bytes = await NetworkUtils.receive_message(reader, timeout=60.0)
                    if weights_bytes:
                        # Load the weights
                        import pickle
                        weights_data = pickle.loads(weights_bytes)
                        partner_state = {k: torch.tensor(v) for k, v in weights_data.items()}
                        self._mix_weights_based_on_fitness(peer_fitness, partner_state)
                        
                        self.logger.info(f"✅ Loaded weights from winner {partner_id}.")
                else:
                    # 3b. PEER: We are the winner, send WINNER response
                    self.logger.info(f"🏆 We are winner. Sending WINNER response to {partner_id}.")
                    await NetworkUtils.send_message(writer, b"RESPONSE|WINNER")

        except asyncio.TimeoutError:
            self.logger.warning(f"⏳ Connection with {partner_id} timed out.")
        except Exception as e:
            self.logger.error(f"Error during gossip with {partner_id}: {e}")
        finally:
            if writer:
                writer.close()
                await writer.wait_closed()
    
    async def start_gossip_protocol(self):
        """Start the gossip protocol"""
        self.gossip_running = True
        
        try:
            # Start server
            self.server = await asyncio.start_server(
                self._handle_peer_connection, 
                '0.0.0.0', 
                self.gossip_port
            )
            
            # Start background tasks
            self.gossip_task = asyncio.create_task(self._gossip_loop())
            # Start the persistent mixer task that listens to the queue
            self.mixer_task = asyncio.create_task(self._mixer())
            
            self.logger.info(f"Node {self.node_id}: Gossip protocol started on port {self.gossip_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start gossip protocol: {e}")
            self.gossip_running = False
    
    async def _gossip_loop(self):
        """Background gossip loop"""
        while self.gossip_running:
            try:
                await self._discover_peers()
                await asyncio.sleep(random.uniform(10, 30))
            except Exception as e:
                self.logger.error(f"Gossip loop error: {e}")
                await asyncio.sleep(30)
    
    async def _mixer(self):
        """A persistent background task that waits for mix requests from the queue."""
        self.logger.info("Mixer task started, waiting for mix requests...")
        
        while self.gossip_running:
            try:
                # This call will wait indefinitely until the training loop puts something in the queue.
                await self.mix_request_queue.get()
                
                # Run the mix initiation logic directly.
                # The lock inside _initiate_mix will prevent overlap with incoming connections.
                await self._initiate_mix()
                
                self.mix_request_queue.task_done()

            except asyncio.CancelledError:
                self.logger.info("Mixer task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in mixer task: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    def request_mix(self):
        """
        A non-blocking, synchronous method for the training loop to request a mix.
        """
        # try_put_nowait is a non-blocking call. If the queue is full (because
        # a mix is already pending), it does nothing and returns immediately.
        try:
            self.mix_request_queue.put_nowait(True)
        except asyncio.QueueFull:
            pass  # A mix is already scheduled, which is fine.
    
    async def _initiate_mix(self):
        """Initiates a mix with a peer using the unified gossip protocol."""
        if not self.peer_list:
            return

        self.mixing_attempts += 1
        
        # Pick a random peer
        partner = self.mixing_rng.choice(list(self.peer_list.keys()))
        host, port = partner.split(':')
        port = int(port)
        
        connection = await NetworkUtils.safe_connect(host, port, timeout=5.0)
        if not connection:
            self.logger.info(f"❌ Could not connect to {partner}")
            return
        
        reader, writer = connection
        
        try:
            # Disable Nagle's algorithm for immediate packet sending
            sock = writer.get_extra_info('socket')
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Use the unified protocol as initiator
            await self._run_gossip_protocol(reader, writer, is_initiator=True)
            
        except Exception as e:
            self.logger.error(f"Error during mix initiation with {partner}: {e}")
    
    async def _discover_peers(self):
        """Discover peers from bootstrap nodes"""
        for bootstrap_addr in self.bootstrap_nodes:
            if bootstrap_addr not in self.peer_list:
                host, port = bootstrap_addr.split(':')
                if host in ['localhost', '127.0.0.1'] and int(port) == self.gossip_port:
                    continue
                    
                self.peer_list[bootstrap_addr] = {
                    'fitness': 1.0,
                    'last_seen': time.time()
                }
    
    def stop_gossip_protocol(self):
        """Stop the gossip protocol"""
        self.gossip_running = False
        
        if self.gossip_task:
            self.gossip_task.cancel()
        if hasattr(self, 'mixer_task') and self.mixer_task:
            self.mixer_task.cancel()
        if self.server:
            self.server.close()
        
        success_rate = (self.successful_mixes / max(1, self.mixing_attempts)) * 100
        self.logger.info(f"Gossip stopped. Success rate: {success_rate:.1f}%")
    
    def get_status(self) -> dict:
        """Get current status"""
        return {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'recent_loss': self.fitness_tracker.get_recent_loss(),
            'peer_count': len(self.peer_list),
            'mixing_attempts': self.mixing_attempts,
            'successful_mixes': self.successful_mixes,
            'pending_mixes': len(self.pending_mixes),
            'step_count': self.step_count
        }

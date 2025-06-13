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
from typing import Dict, List, Optional, Set
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module, 
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_interval: int = 100):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        
        # Per-process random state
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_interval = mixing_interval
        self.is_mixing = False  # State flag to prevent concurrent mixes
        
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
    
    async def attempt_mix_if_scheduled(self, step: int):
        """
        Checks if it's time to mix based on a fixed interval and if a mix is not
        already in progress. If so, it starts the mixing process.
        """
        # 1. Gating: Don't start a new mix if one is already happening.
        if self.is_mixing:
            return

        # 2. Scheduling: Only mix on the specified step interval.
        if step > 0 and step % self.mixing_interval == 0:
            await self._initiate_mix()
    
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
        """Handle incoming connections"""
        try:
            client_addr = writer.get_extra_info('peername', 'unknown')
            
            # Try to read the first part of the message to determine type
            data = await reader.read(256)
            if not data:
                return
            
            message = data.decode().strip()
            
            if message.startswith("FITNESS_PROBE"):
                # Reset reader to beginning and handle fitness probe
                await self._handle_fitness_probe(reader, writer)
            else:
                # Try to parse as JSON for backward compatibility
                try:
                    request = json.loads(message)
                    msg_type = request.get('type')
                    
                    if msg_type == 'mixing_proposal':
                        await self._handle_mixing_proposal(reader, writer, request)
                    else:
                        self.logger.warning(f"Unknown message type: {msg_type}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Unknown message format: {message[:50]}")
                
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except:
                pass
    
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
    
    async def _handle_fitness_probe(self, reader, writer):
        """
        Handles an incoming fitness probe from a peer and decides whether to
        request their weights if they are the winner.
        """
        peer_addr = writer.get_extra_info('peername')
        try:
            # 1. Receive the fitness probe
            data = (await reader.read(256)).decode().strip()
            parts = data.split('|')
            if not data.startswith("FITNESS_PROBE") or len(parts) != 3:
                self.logger.error(f"Invalid probe from {peer_addr}: {data}")
                return

            peer_id, peer_fitness_str = parts[1], parts[2]
            peer_fitness = float(peer_fitness_str)
            current_fitness = self.get_current_fitness()
            
            self.logger.info(f"📬 Received fitness probe from {peer_id} (fitness: {peer_fitness:.4f}). Our fitness: {current_fitness:.4f}")

            # 2. Compare fitness. If the peer is better, request their weights.
            if peer_fitness > current_fitness:
                self.logger.info(f"🥈 We are the loser. Requesting weights from {peer_id}.")
                
                # a) Send the request
                writer.write("REQUEST_WEIGHTS".encode())
                await writer.drain()

                # b) Receive the weights
                num_bytes_header = await asyncio.wait_for(reader.readexactly(8), timeout=15.0)
                num_bytes = int.from_bytes(num_bytes_header, 'big')
                
                self.logger.info(f"Receiving {num_bytes} bytes from winner {peer_id}...")
                winner_weights_bytes = await asyncio.wait_for(reader.readexactly(num_bytes), timeout=60.0)
                
                # Load the weights
                import pickle
                weights_data = pickle.loads(winner_weights_bytes)
                partner_state = {k: torch.tensor(v) for k, v in weights_data.items()}
                self._mix_weights_based_on_fitness(peer_fitness, partner_state)
                
                self.logger.info(f"✅ Successfully received and loaded weights from {peer_id}.")
            else:
                # 3. If we are the winner, do nothing and close the connection.
                self.logger.info(f"🏆 We are the winner. Closing connection with {peer_id}.")
                writer.write("NO_THANKS".encode())
                await writer.drain()

        except Exception as e:
            self.logger.error(f"Error in handle_fitness_probe with {peer_addr}: {e}")
        finally:
            # Close the connection after handling
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
            # The discovery task has been removed as it is redundant
            
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
    
    # The redundant _discovery_loop background task has been removed.
    # The logic for initiating mixes is handled by `check_scheduled_mixing`
    # and the logic for receiving mixes is handled by the asyncio server.
    
    async def _initiate_mix(self):
        """
        Initiates a mix with a peer by sending its fitness and waiting for a
        potential request to send its weights back.
        """
        if not self.peer_list:
            return

        self.is_mixing = True  # Set lock: A mix is now in progress.
        try:
            # Pick a random peer
            partner = self.mixing_rng.choice(list(self.peer_list.keys()))
            
            current_fitness = self.get_current_fitness()
            recent_loss = self.fitness_tracker.get_recent_loss()
            
            self.logger.info(f"🎲 INITIATING MIXING: partner={partner}")
            self.logger.info(f"  Current fitness: {current_fitness:.4f}, Recent loss: {recent_loss:.4f}")
            
            host, port = partner.split(':')
            port = int(port)
            
            connection = await NetworkUtils.safe_connect(host, port, timeout=5.0)
            if not connection:
                self.logger.info(f"❌ Could not connect to {partner}")
                return
            
            reader, writer = connection
            
            try:
                # --- CRITICAL: Disable Nagle's algorithm ---
                # This ensures our small probe message is sent immediately without being
                # buffered by the OS, which prevents the peer from timing out.
                sock = writer.get_extra_info('socket')
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # -----------------------------------------

                # 1. Send fitness probe
                request = f"FITNESS_PROBE|{self.node_id}|{current_fitness}"
                writer.write(request.encode())
                await writer.drain()
                
                # 2. Wait for response from peer
                response = (await asyncio.wait_for(reader.read(100), timeout=10.0)).decode()
                
                # 3. If peer requests our weights, send them
                if response == "REQUEST_WEIGHTS":
                    self.logger.info(f"🏆 Peer {partner} is loser, sending our weights.")
                    
                    # Get model weights as bytes
                    state_dict = self.model.state_dict()
                    weights_data = {}
                    for k, v in state_dict.items():
                        weights_data[k] = v.cpu().numpy().tolist()
                    
                    import pickle
                    weights_bytes = pickle.dumps(weights_data)
                    
                    # Send weights
                    writer.write(len(weights_bytes).to_bytes(8, 'big'))
                    writer.write(weights_bytes)
                    await writer.drain()
                    
                    self.successful_mixes += 1
                    self.logger.info(f"✅ Successfully sent weights to {partner}.")
                else:
                    self.logger.info(f"🥈 Peer {partner} is winner, ending mix.")
                
                self.mixing_attempts += 1
                
            finally:
                writer.close()
                await writer.wait_closed()
                
        except asyncio.TimeoutError:
            self.logger.warning(f"⏳ Mix with {partner} timed out.")
            self.mixing_attempts += 1
        except Exception as e:
            self.logger.error(f"Error during mix initiation: {e}")
            self.mixing_attempts += 1
        finally:
            self.is_mixing = False  # Release lock: The mix is done (success or fail).
    
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
        # Discovery task no longer exists
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

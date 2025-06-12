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
from typing import Dict, List, Optional
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module, 
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_frequency: float = 0.02):  # 2% chance per step
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        
        # FIXED: Per-process random state
        self.mixing_rng = random.Random(42 + global_rank * 1000)  # Unique seed per rank
        
        # CHANGED: Frequency-based mixing instead of interval
        self.mixing_frequency = mixing_frequency  # Probability per step
        
        # Fitness tracking
        self.fitness_tracker = FitnessTracker()
        self.step_count = 0
        
        # Peer management
        self.peer_list: Dict[str, dict] = {}
        
        # Track if we're currently mixing (prevent concurrent mixing)
        self.currently_mixing = False
        
        # Network setup - SMART PORT ASSIGNMENT
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.gossip_port = NetworkUtils.get_gossip_port(data_parallel_rank, master_addr)
        self.server = None
        self.gossip_running = False
        self.gossip_task = None
        
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
    
    def should_attempt_mixing(self) -> bool:
        """Frequency-based mixing decision - check every step"""
        if self.currently_mixing:
            self.logger.info(f"Already mixing, skipping attempt")
            return False
            
        if len(self.peer_list) == 0:
            return False
            
        # Simple frequency check with per-process randomness
        will_mix = self.mixing_rng.random() < self.mixing_frequency
        
        if will_mix:
            self.logger.info(f"🎲 Random mixing trigger! (freq={self.mixing_frequency}, peers={len(self.peer_list)})")
        
        return will_mix
    
    def select_mixing_partner(self) -> Optional[str]:
        """Simplified partner selection - just pick randomly from available peers"""
        if not self.peer_list:
            return None
        
        # Simple random selection from peers
        peer_ids = list(self.peer_list.keys())
        selected = self.mixing_rng.choice(peer_ids)
        
        self.logger.info(f"Randomly selected {selected} from {len(peer_ids)} peers")
        return selected
    
    def update_peer_fitness(self, peer_id: str, fitness: float):
        """Update fitness information for a peer"""
        if peer_id in self.peer_list:
            self.peer_list[peer_id]['fitness'] = fitness
            self.peer_list[peer_id]['last_seen'] = time.time()
    
    async def attempt_weight_mixing(self, training_step: Optional[int] = None):
        """Simplified mixing attempt - frequency based"""
        self.logger.info(f">>> attempt_weight_mixing() called at training step {training_step}")
        
        if not self.should_attempt_mixing():
            return False
            
        partner_id = self.select_mixing_partner()
        if not partner_id:
            self.logger.info(f">>> No mixing partner selected")
            return False
        
        self.logger.info(f">>> Selected partner: {partner_id}")
        
        # Set mixing flag
        self.currently_mixing = True
        
        try:
            self.mixing_attempts += 1
            success = await self._negotiate_weight_mixing(partner_id)
            
            if success:
                self.successful_mixes += 1
                self.logger.info(f"✅ Mixing successful with {partner_id}")
            else:
                self.logger.info(f"❌ Mixing failed with {partner_id}")
                
            return success
            
        finally:
            # Always clear mixing flag
            self.currently_mixing = False
    
    async def _negotiate_weight_mixing(self, partner_id: str) -> bool:
        """Negotiate weight mixing with a peer - enhanced error handling"""
        current_fitness = self.get_current_fitness()
        
        try:
            # Parse partner address
            if ':' not in partner_id:
                self.logger.error(f"Invalid partner address format: {partner_id}")
                return False
                
            host, port = partner_id.split(':')
            port = int(port)
            
            self.logger.info(f"🔌 Connecting to {partner_id}")
            
            # Connect to partner with timeout
            connection = await NetworkUtils.safe_connect(host, port, timeout=5.0)
            if not connection:
                self.logger.error(f"Failed to connect to {partner_id}")
                return False
                
            reader, writer = connection
            
            try:
                # Send mixing proposal
                proposal = {
                    'type': 'mixing_proposal',
                    'sender': self.node_id,
                    'fitness': current_fitness,
                    'model_hash': self._get_model_hash(),
                    'timestamp': time.time()
                }
                
                self.logger.info(f"📤 Sending proposal to {partner_id}")
                if not await NetworkUtils.safe_send_json(writer, proposal, timeout=5.0):
                    self.logger.error(f"Failed to send proposal to {partner_id}")
                    return False
                
                # Wait for response
                self.logger.info(f"⏳ Waiting for response from {partner_id}")
                response = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
                if not response:
                    self.logger.error(f"No response from {partner_id}")
                    return False
                    
                if not response.get('accept'):
                    self.logger.info(f"❌ Proposal rejected by {partner_id}")
                    return False
                
                self.logger.info(f"✅ Proposal accepted by {partner_id}")
                
                # Perform actual weight mixing
                success = await self._perform_weight_mixing(writer, reader, response)
                
                if success:
                    self.logger.info(f"🎯 Successfully mixed with {partner_id}, fitness={current_fitness:.4f}")
                else:
                    self.logger.error(f"❌ Weight mixing failed with {partner_id}")
                
                return success
                
            finally:
                # Always close the connection
                try:
                    if not writer.is_closing():
                        writer.close()
                        await writer.wait_closed()
                except Exception as close_error:
                    self.logger.error(f"Error closing connection to {partner_id}: {close_error}")
                
        except ValueError as e:
            self.logger.error(f"Invalid port in partner address {partner_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Weight mixing negotiation failed with {partner_id}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def _perform_weight_mixing(self, writer, reader, partner_info) -> bool:
        """Actually mix the weights"""
        try:
            current_fitness = self.get_current_fitness()
            partner_fitness = partner_info['fitness']
            
            # Adaptive mixing ratio based on fitness
            if partner_fitness > current_fitness:
                alpha = 0.7  # Take more from better partner
            else:
                alpha = 0.3  # Conservative mixing
                
            # Add randomness to avoid determinism
            alpha += random.uniform(-0.1, 0.1)
            alpha = np.clip(alpha, 0.1, 0.9)
            
            # Request partner weights
            request = {'type': 'weight_request'}
            if not await NetworkUtils.safe_send_json(writer, request):
                return False
            
            # Receive partner weights (simplified - in practice use streaming)
            weight_response = await NetworkUtils.safe_recv_json(reader)
            if not weight_response or weight_response.get('type') != 'weight_data':
                return False
            
            # For safety, we'll do a simplified mixing
            # In full implementation, you'd stream actual weight tensors
            self._apply_fitness_perturbation(alpha, partner_fitness)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Weight mixing failed: {e}")
            return False
    
    def _apply_fitness_perturbation(self, alpha: float, partner_fitness: float):
        """Apply fitness-based perturbation (simplified mixing)"""
        current_fitness = self.get_current_fitness()
        
        # Fitness-based perturbation scale
        if partner_fitness > current_fitness:
            # Partner is better - larger exploration
            perturbation_scale = 0.001 * alpha
        else:
            # We're better - smaller perturbation
            perturbation_scale = 0.0005 * alpha
        
        with torch.no_grad():
            for param in self.model.parameters():
                if random.random() < 0.1:  # Only touch 10% of parameters
                    noise = torch.randn_like(param) * perturbation_scale
                    param.data += noise
    
    def _get_model_hash(self) -> str:
        """Get hash of model weights for verification"""
        model_str = ""
        for name, param in self.model.named_parameters():
            model_str += param.data.cpu().numpy().tobytes().hex()[:100]  # Truncate for speed
        return hashlib.md5(model_str.encode()).hexdigest()[:16]
    
    async def _handle_peer_connection(self, reader, writer):
        """Handle incoming peer connections with robust error handling"""
        client_addr = "unknown"
        try:
            # Get client address for debugging
            client_addr = writer.get_extra_info('peername', 'unknown')
            self.logger.info(f"📞 Incoming connection from {client_addr}")
            
            # Receive request with timeout
            request = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
            if not request:
                self.logger.warning(f"No request received from {client_addr}")
                return
            
            self.logger.info(f"📨 Received {request.get('type', 'unknown')} from {client_addr}")
            
            if request['type'] == 'mixing_proposal':
                await self._handle_mixing_proposal(reader, writer, request)
            elif request['type'] == 'peer_exchange':
                # We don't actually implement peer exchange yet, just acknowledge
                self.logger.info(f"Peer exchange not implemented, ignoring")
            else:
                self.logger.warning(f"Unknown request type: {request.get('type')}")
                
        except KeyError as e:
            self.logger.error(f"Missing key in request from {client_addr}: {e}")
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout handling connection from {client_addr}")
        except ConnectionResetError:
            self.logger.info(f"Connection reset by {client_addr}")
        except Exception as e:
            self.logger.error(f"Error handling peer connection from {client_addr}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            # Robust cleanup
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
                self.logger.info(f"✅ Connection to {client_addr} closed cleanly")
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup for {client_addr}: {cleanup_error}")
    
    async def _handle_mixing_proposal(self, reader, writer, proposal):
        """Handle incoming mixing proposal with robust error handling"""
        partner_id = proposal.get('sender', 'unknown')
        
        try:
            partner_fitness = proposal['fitness']
            current_fitness = self.get_current_fitness()
            
            # Log incoming proposal
            self.logger.info(f"📨 MIXING PROPOSAL: {partner_id} -> Node {self.node_id}")
            self.logger.info(f"   Partner fitness: {partner_fitness:.6f}, My fitness: {current_fitness:.6f}")
            
            # Update peer fitness
            self.update_peer_fitness(partner_id, partner_fitness)
            
            # SIMPLIFIED DECISION: Accept unless we're already busy
            accept = not self.currently_mixing
            
            if accept:
                reason = "Accepting incoming proposal"
                self.currently_mixing = True  # Set busy flag
            else:
                reason = "Rejecting - already mixing"
            
            self.logger.info(f"   Decision: {'ACCEPT' if accept else 'REJECT'} - {reason}")
            
            response = {
                'accept': accept,
                'fitness': current_fitness,
                'sender': self.node_id  # Include our node_id
            }
            
            # Send response with error checking
            if not await NetworkUtils.safe_send_json(writer, response, timeout=5.0):
                self.logger.error(f"Failed to send response to {partner_id}")
                return
            
            if accept:
                try:
                    # Handle weight exchange
                    self.logger.info(f"Waiting for weight request from {partner_id}")
                    weight_request = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
                    
                    if weight_request and weight_request.get('type') == 'weight_request':
                        # Send simplified weight data
                        weight_response = {
                            'type': 'weight_data',
                            'fitness': current_fitness,
                            'hash': self._get_model_hash()
                        }
                        
                        if await NetworkUtils.safe_send_json(writer, weight_response, timeout=5.0):
                            self.logger.info(f"✅ Completed weight exchange with {partner_id}")
                        else:
                            self.logger.error(f"Failed to send weight data to {partner_id}")
                    else:
                        self.logger.warning(f"Invalid or missing weight request from {partner_id}")
                        
                except Exception as exchange_error:
                    self.logger.error(f"Error during weight exchange with {partner_id}: {exchange_error}")
                finally:
                    # Always clear busy flag
                    self.currently_mixing = False
                    
        except KeyError as e:
            self.logger.error(f"Missing required field in proposal from {partner_id}: {e}")
        except Exception as e:
            self.logger.error(f"Error handling mixing proposal from {partner_id}: {e}")
            # Make sure to clear busy flag on any error
            self.currently_mixing = False
    
    async def start_gossip_protocol(self):
        """Start the gossip protocol"""
        self.gossip_running = True
        
        try:
            # Start server for incoming connections
            self.server = await asyncio.start_server(
                self._handle_peer_connection, 
                '0.0.0.0', 
                self.gossip_port
            )
            
            # Start continuous gossip loop
            self.gossip_task = asyncio.create_task(self._gossip_loop())
            
            self.logger.info(f"Node {self.node_id}: Gossip protocol started on port {self.gossip_port}")
            self.logger.info(f"Bootstrap nodes: {self.bootstrap_nodes}")
            
        except Exception as e:
            self.logger.error(f"Failed to start gossip protocol: {e}")
            self.gossip_running = False
    
    async def _gossip_loop(self):
        """Main gossip loop - runs continuously"""
        while self.gossip_running:
            try:
                await asyncio.gather(
                    self._discover_peers(),
                    self._cleanup_dead_peers(),
                    return_exceptions=True
                )
                # Jitter to avoid synchronization
                await asyncio.sleep(random.uniform(10, 30))
            except Exception as e:
                self.logger.error(f"Gossip loop error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _discover_peers(self):
        """Discover peers from bootstrap nodes"""
        for bootstrap_addr in self.bootstrap_nodes:
            if bootstrap_addr not in self.peer_list:
                # Don't add self
                host, port = bootstrap_addr.split(':')
                if host == 'localhost' and int(port) == self.gossip_port:
                    self.logger.info(f"Skipping self address: {bootstrap_addr} (my port: {self.gossip_port})")
                    continue
                    
                self.peer_list[bootstrap_addr] = {
                    'fitness': 1.0,
                    'last_seen': time.time(),
                    'address': bootstrap_addr
                }
                self.logger.info(f"Added peer: {bootstrap_addr}")
        
        # Log current peer status
        if len(self.peer_list) > 0:
            self.logger.info(f"Current peer list: {list(self.peer_list.keys())}")
        else:
            self.logger.warning(f"No peers discovered! Bootstrap nodes: {self.bootstrap_nodes}, My port: {self.gossip_port}")
    
    async def _cleanup_dead_peers(self):
        """Remove peers that haven't been seen recently"""
        current_time = time.time()
        dead_peers = []
        
        for peer_id, peer_data in self.peer_list.items():
            if current_time - peer_data.get('last_seen', 0) > 600:  # 10 minutes
                dead_peers.append(peer_id)
        
        for peer_id in dead_peers:
            del self.peer_list[peer_id]
            self.logger.info(f"Removed dead peer: {peer_id}")
    
    def stop_gossip_protocol(self):
        """Stop the gossip protocol"""
        self.gossip_running = False
        
        if self.gossip_task:
            self.gossip_task.cancel()
            
        if self.server:
            self.server.close()
            
        # Log statistics
        success_rate = (self.successful_mixes / max(1, self.mixing_attempts)) * 100
        self.logger.info(f"Node {self.node_id}: Gossip stopped. Mix success rate: {success_rate:.1f}%")
    
    def get_status(self) -> dict:
        """Get current status for debugging"""
        return {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'recent_loss': self.fitness_tracker.get_recent_loss(),
            'peer_count': len(self.peer_list),
            'mixing_attempts': self.mixing_attempts,
            'successful_mixes': self.successful_mixes,
            'step_count': self.step_count
        }

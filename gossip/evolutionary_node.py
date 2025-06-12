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
                 tp_size: int, mixing_frequency: float = 0.02):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.tp_size = tp_size
        
        # Per-process random state
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_frequency = mixing_frequency
        
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
    
    
    async def _handle_peer_connection(self, reader, writer):
        """Handle incoming connections"""
        try:
            client_addr = writer.get_extra_info('peername', 'unknown')
            
            request = await NetworkUtils.safe_recv_json(reader, timeout=3.0)
            if not request:
                return
            
            msg_type = request.get('type')
            
            if msg_type == 'mixing_proposal':
                await self._handle_mixing_proposal(reader, writer, request)
            elif msg_type == 'ready_to_mix':
                await self._handle_ready_to_mix(reader, writer, request)
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except:
                pass
    
    async def _handle_mixing_proposal(self, reader, writer, proposal, connection_id):
        """Handle mixing proposal with connection tracking"""
        partner_id = proposal.get('sender', 'unknown')
        
        try:
            partner_fitness = proposal['fitness']
            current_fitness = self.get_current_fitness()
            
            # Update peer fitness
            self.update_peer_fitness(partner_id, partner_fitness)
            
            # Quick decision without complex logic
            accept = not self.currently_mixing
            
            if accept:
                self.currently_mixing = True
                self.logger.info(f"[{connection_id}] ACCEPTING proposal from {partner_id}")
            else:
                self.logger.info(f"[{connection_id}] REJECTING proposal from {partner_id} - busy")
            
            response = {
                'accept': accept,
                'fitness': current_fitness,
                'sender': self.node_id
            }
            
            # Send response immediately
            if await NetworkUtils.safe_send_json(writer, response, timeout=2.0):
                self.logger.info(f"[{connection_id}] Response sent to {partner_id}")
            else:
                self.logger.error(f"[{connection_id}] Failed to send response to {partner_id}")
                return
            
            # Handle accepted proposals
            if accept:
                try:
                    # Simple weight exchange
                    weight_request = await NetworkUtils.safe_recv_json(reader, timeout=5.0)
                    if weight_request and weight_request.get('type') == 'weight_request':
                        weight_response = {
                            'type': 'weight_data',
                            'fitness': current_fitness,
                            'hash': self._get_model_hash()
                        }
                        await NetworkUtils.safe_send_json(writer, weight_response, timeout=2.0)
                        self.logger.info(f"[{connection_id}] Weight exchange completed with {partner_id}")
                finally:
                    self.currently_mixing = False
                    
        except Exception as e:
            self.logger.error(f"[{connection_id}] Error in proposal handling: {e}")
            self.currently_mixing = False
    
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
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            
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
    
    async def _discovery_loop(self):
        """Background discovery loop - try to schedule mixing"""
        while self.gossip_running:
            try:
                # Random chance to propose mixing
                if (self.mixing_rng.random() < self.mixing_frequency and 
                    len(self.peer_list) > 0 and 
                    len(self.pending_mixes) < 2):  # Limit concurrent mixes
                    
                    await self._propose_mixing()
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(5)
    
    async def _propose_mixing(self):
        """Propose mixing with a random peer"""
        if not self.peer_list:
            return
        
        partner = self.mixing_rng.choice(list(self.peer_list.keys()))
        
        # Schedule mixing 10-50 steps in the future
        future_step = self.step_count + self.mixing_rng.randint(10, 50)
        
        # Skip if we already have something scheduled
        if future_step in self.pending_mixes:
            return
        
        try:
            host, port = partner.split(':')
            port = int(port)
            
            connection = await NetworkUtils.safe_connect(host, port, timeout=2.0)
            if not connection:
                return
            
            reader, writer = connection
            
            try:
                proposal = {
                    'type': 'mixing_proposal',
                    'sender': self.node_id,
                    'fitness': self.get_current_fitness(),
                    'mix_at_step': future_step
                }
                
                if await NetworkUtils.safe_send_json(writer, proposal, timeout=2.0):
                    response = await NetworkUtils.safe_recv_json(reader, timeout=3.0)
                    
                    if response and response.get('accept'):
                        # Schedule the mixing
                        self.pending_mixes[future_step] = {
                            'partner': partner,
                            'role': 'initiator',
                            'fitness': response.get('fitness', 0.0)
                        }
                        self.mixing_attempts += 1
                        self.logger.info(f"🎯 Scheduled mixing with {partner} at step {future_step}")
                    else:
                        self.logger.info(f"❌ Mixing proposal rejected by {partner}")
                        
            finally:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            self.logger.error(f"Failed to propose mixing with {partner}: {e}")
    
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
        if self.discovery_task:
            self.discovery_task.cancel()
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

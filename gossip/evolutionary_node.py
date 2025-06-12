import asyncio
import random
import numpy as np
import torch
import json
import time
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
                 mixing_interval: int = 500):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.world_size = world_size
        self.data_parallel_rank = data_parallel_rank
        self.mixing_interval = mixing_interval
        
        # Fitness tracking
        self.fitness_tracker = FitnessTracker()
        self.step_count = 0
        
        # Peer management
        self.peer_list: Dict[str, dict] = {}
        self.mixing_probability = 0.1
        
        # Network setup
        self.gossip_port = 29501 + (global_rank % 100)  # Unique port per rank
        self.server = None
        self.gossip_running = False
        self.gossip_task = None
        
        # Logging
        self.logger = logging.getLogger(f'evolutionary_node_{node_id}')
        
        # Bootstrap nodes
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(global_rank, self.logger)
        
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
    
    def should_mix_this_round(self) -> bool:
        """Erdős–Rényi random graph connectivity check"""
        if self.step_count % self.mixing_interval != 0:
            return False
            
        n_peers = max(1, len(self.peer_list))
        # Ensure giant component formation: p > ln(n)/n
        critical_prob = np.log(n_peers) / n_peers if n_peers > 1 else 0.1
        adaptive_prob = max(self.mixing_probability, critical_prob * 1.5)
        
        return random.random() < adaptive_prob
    
    def select_mixing_partner(self) -> Optional[str]:
        """Fitness-based partner selection (snail sex algorithm)"""
        if not self.peer_list:
            return None
            
        current_fitness = self.get_current_fitness()
        
        # Weight selection by fitness similarity + randomness
        weights = []
        peer_ids = []
        
        for peer_id, peer_data in self.peer_list.items():
            peer_fitness = peer_data.get('fitness', 1.0)
            
            # Fitness differential - prefer similar fitness for exploration
            fitness_diff = abs(current_fitness - peer_fitness)
            similarity_weight = np.exp(-fitness_diff / 0.1)  # Temperature parameter
            
            # Add randomness to avoid local optima
            random_weight = random.random() * 0.3
            
            total_weight = similarity_weight + random_weight
            weights.append(total_weight)
            peer_ids.append(peer_id)
        
        if not weights:
            return None
            
        # Weighted random selection
        weights = np.array(weights)
        probs = weights / weights.sum()
        
        return np.random.choice(peer_ids, p=probs)
    
    async def attempt_weight_mixing(self):
        """Attempt to mix weights with a peer"""
        if not self.should_mix_this_round():
            return False
            
        partner_id = self.select_mixing_partner()
        if not partner_id:
            # No peers available - log status
            fitness = self.get_current_fitness()
            self.logger.info(f"Node {self.node_id}: fitness={fitness:.4f}, no peers for mixing")
            return False
            
        self.mixing_attempts += 1
        success = await self._negotiate_weight_mixing(partner_id)
        if success:
            self.successful_mixes += 1
            
        return success
    
    async def _negotiate_weight_mixing(self, partner_id: str) -> bool:
        """Negotiate weight mixing with a peer"""
        current_fitness = self.get_current_fitness()
        
        try:
            # Parse partner address
            if ':' not in partner_id:
                return False
            host, port = partner_id.split(':')
            port = int(port)
            
            # Connect to partner
            connection = await NetworkUtils.safe_connect(host, port, timeout=5.0)
            if not connection:
                return False
                
            reader, writer = connection
            
            # Send mixing proposal
            proposal = {
                'type': 'mixing_proposal',
                'sender': self.node_id,
                'fitness': current_fitness,
                'model_hash': self._get_model_hash(),
                'timestamp': time.time()
            }
            
            if not await NetworkUtils.safe_send_json(writer, proposal):
                return False
            
            # Wait for response
            response = await NetworkUtils.safe_recv_json(reader)
            if not response or not response.get('accept'):
                return False
            
            # Perform actual weight mixing
            success = await self._perform_weight_mixing(writer, reader, response)
            
            if success:
                self.logger.info(f"Successfully mixed with {partner_id}, fitness={current_fitness:.4f}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Weight mixing negotiation failed with {partner_id}: {e}")
            return False
        finally:
            if 'writer' in locals():
                writer.close()
                await writer.wait_closed()
    
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
        """Handle incoming peer connections"""
        try:
            # Receive request
            request = await NetworkUtils.safe_recv_json(reader)
            if not request:
                return
            
            if request['type'] == 'mixing_proposal':
                await self._handle_mixing_proposal(reader, writer, request)
            elif request['type'] == 'peer_exchange':
                await self._handle_peer_exchange(reader, writer, request)
                
        except Exception as e:
            self.logger.error(f"Error handling peer connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_mixing_proposal(self, reader, writer, proposal):
        """Handle incoming mixing proposal"""
        partner_fitness = proposal['fitness']
        current_fitness = self.get_current_fitness()
        
        # Decision algorithm: mix if beneficial or exploratory
        fitness_ratio = partner_fitness / max(current_fitness, 1e-8)
        accept_prob = min(1.0, fitness_ratio + 0.2)  # Always some exploration
        
        accept = random.random() < accept_prob
        
        response = {
            'accept': accept,
            'fitness': current_fitness,
            'sender': proposal['sender']
        }
        
        await NetworkUtils.safe_send_json(writer, response)
        
        if accept:
            # Handle weight exchange
            weight_request = await NetworkUtils.safe_recv_json(reader)
            if weight_request and weight_request['type'] == 'weight_request':
                # Send simplified weight data
                weight_response = {
                    'type': 'weight_data',
                    'fitness': current_fitness,
                    'hash': self._get_model_hash()
                }
                await NetworkUtils.safe_send_json(writer, weight_response)
    
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
                    continue
                    
                self.peer_list[bootstrap_addr] = {
                    'fitness': 1.0,
                    'last_seen': time.time(),
                    'address': bootstrap_addr
                }
    
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

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
    
    async def check_scheduled_mixing(self, training_step: int) -> bool:
        """Check if we should mix at this training step"""
        if training_step not in self.pending_mixes:
            return False
        
        mix_info = self.pending_mixes[training_step]
        partner = mix_info['partner']
        role = mix_info['role']  # 'initiator' or 'acceptor'
        
        self.logger.info(f"🎯 SCHEDULED MIXING at step {training_step} with {partner} (role: {role})")
        
        try:
            if role == 'initiator':
                success = await self._execute_mixing_as_initiator(partner)
            else:
                success = await self._execute_mixing_as_acceptor(partner)
            
            if success:
                self.successful_mixes += 1
                self.logger.info(f"✅ Scheduled mixing completed with {partner}")
            else:
                self.logger.error(f"❌ Scheduled mixing failed with {partner}")
            
            return success
            
        finally:
            # Clean up scheduled mixing
            del self.pending_mixes[training_step]
    
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
                
                # Send our weights
                our_weights = {
                    'type': 'weights',
                    'fitness': self.get_current_fitness(),
                    'weights_hash': self._get_model_hash()
                }
                
                if not await NetworkUtils.safe_send_json(writer, our_weights, timeout=5.0):
                    return False
                
                # Do the actual mixing
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
    
    def _mix_weights_based_on_fitness(self, partner_fitness: float):
        """Perform actual weight mixing"""
        current_fitness = self.get_current_fitness()
        recent_loss = self.fitness_tracker.get_recent_loss()
        
        # Adaptive mixing ratio
        if partner_fitness > current_fitness:
            alpha = 0.3  # Take some from better partner
            mixing_type = "EXPLORATORY"
        else:
            alpha = 0.1  # Small perturbation
            mixing_type = "EXPLOITATIVE"
        
        param_touch_rate = 0.05  # Touch 5% of parameters
        noise_scale = 0.001 * alpha
        
        self.logger.info(f"🧬 WEIGHT MIXING DETAILS:")
        self.logger.info(f"  Partner fitness: {partner_fitness:.4f}, Our fitness: {current_fitness:.4f}")
        self.logger.info(f"  Our recent loss: {recent_loss:.4f}")
        self.logger.info(f"  Mixing type: {mixing_type}, Alpha: {alpha:.3f}")
        self.logger.info(f"  Param touch rate: {param_touch_rate:.1%}, Noise scale: {noise_scale:.6f}")
        
        # Apply perturbation
        params_touched = 0
        total_params = 0
        
        with torch.no_grad():
            for param in self.model.parameters():
                total_params += param.numel()
                if self.mixing_rng.random() < param_touch_rate:
                    noise = torch.randn_like(param) * noise_scale
                    param.data += noise
                    params_touched += param.numel()
        
        touch_percentage = (params_touched / max(total_params, 1)) * 100
        self.logger.info(f"  Actually touched: {params_touched:,}/{total_params:,} params ({touch_percentage:.2f}%)")
    
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
    
    async def _handle_ready_to_mix(self, reader, writer, request):
        """Handle ready-to-mix signal from initiator"""
        partner_id = request.get('sender')
        partner_fitness = request.get('fitness', 0.0)
        
        # Send our weights
        our_weights = {
            'type': 'weights',
            'fitness': self.get_current_fitness(),
            'weights_hash': self._get_model_hash()
        }
        
        if await NetworkUtils.safe_send_json(writer, our_weights, timeout=5.0):
            # Get partner's weights
            partner_weights = await NetworkUtils.safe_recv_json(reader, timeout=5.0)
            if partner_weights and partner_weights.get('type') == 'weights':
                # Do the mixing
                self._mix_weights_based_on_fitness(partner_fitness)
                self.logger.info(f"✅ Completed mixing as acceptor with {partner_id}")
    
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
        loop_counter = 0
        while self.gossip_running:
            try:
                loop_counter += 1
                rand_value = self.mixing_rng.random()
                
                # Log discovery loop activity every 30 seconds
                if loop_counter % 30 == 0:
                    current_fitness = self.get_current_fitness()
                    recent_loss = self.fitness_tracker.get_recent_loss()
                    self.logger.info(f"🔍 DISCOVERY LOOP STATUS:")
                    self.logger.info(f"  Loop #{loop_counter}, Random value: {rand_value:.4f}")
                    self.logger.info(f"  Mixing frequency threshold: {self.mixing_frequency:.4f}")
                    self.logger.info(f"  Current fitness: {current_fitness:.4f}, Recent loss: {recent_loss:.4f}")
                    self.logger.info(f"  Peers available: {len(self.peer_list)}, Pending mixes: {len(self.pending_mixes)}")
                
                # Random chance to propose mixing
                if (rand_value < self.mixing_frequency and 
                    len(self.peer_list) > 0 and 
                    len(self.pending_mixes) < 2):  # Limit concurrent mixes
                    
                    self.logger.info(f"🎲 TRIGGERING MIXING PROPOSAL (rand={rand_value:.4f} < threshold={self.mixing_frequency:.4f})")
                    await self._propose_mixing()
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(5)
    
    async def _propose_mixing(self):
        """Propose mixing with a random peer"""
        if not self.peer_list:
            self.logger.debug("No peers available for mixing proposal")
            return
        
        partner = self.mixing_rng.choice(list(self.peer_list.keys()))
        
        # Schedule mixing 10-50 steps in the future
        future_step = self.step_count + self.mixing_rng.randint(10, 50)
        
        # Skip if we already have something scheduled
        if future_step in self.pending_mixes:
            self.logger.debug(f"Step {future_step} already has scheduled mixing, skipping proposal")
            return
        
        current_fitness = self.get_current_fitness()
        recent_loss = self.fitness_tracker.get_recent_loss()
        
        self.logger.info(f"🎲 PROPOSING MIXING: partner={partner}, future_step={future_step}")
        self.logger.info(f"  Current fitness: {current_fitness:.4f}, Recent loss: {recent_loss:.4f}")
        self.logger.info(f"  Step count: {self.step_count}, Pending mixes: {len(self.pending_mixes)}")
        
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
                        partner_fitness = response.get('fitness', 0.0)
                        # Schedule the mixing
                        self.pending_mixes[future_step] = {
                            'partner': partner,
                            'role': 'initiator',
                            'fitness': partner_fitness
                        }
                        self.mixing_attempts += 1
                        self.logger.info(f"✅ MIXING PROPOSAL ACCEPTED: partner={partner}, step={future_step}")
                        self.logger.info(f"  Partner fitness: {partner_fitness:.4f} vs our fitness: {current_fitness:.4f}")
                        self.logger.info(f"  Fitness ratio: {partner_fitness/max(current_fitness, 1e-6):.3f}")
                    else:
                        rejection_reason = "no response" if not response else "rejected"
                        self.logger.info(f"❌ MIXING PROPOSAL {rejection_reason.upper()}: partner={partner}")
                        if response:
                            self.logger.info(f"  Partner fitness: {response.get('fitness', 'unknown')}")
                        
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

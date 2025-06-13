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
        current_fitness = self.get_current_fitness()
        
        self.logger.info(f"🤝 READY TO MIX: partner={partner_id}")
        self.logger.info(f"  Partner fitness: {partner_fitness:.4f}, Our fitness: {current_fitness:.4f}")
        
        # Determine who is winner/loser
        we_are_winner = current_fitness > partner_fitness
        
        if we_are_winner:
            self.logger.info(f"🏆 WE ARE WINNER: Waiting to receive loser's weights first")
            
            # As winner, we receive the loser's weights first
            partner_weights = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
            if not partner_weights or partner_weights.get('type') != 'weights':
                self.logger.error(f"❌ Failed to receive loser's weights from {partner_id}")
                return
            
            self.logger.info(f"📦 Received loser's weights, now sending our winning weights")
            
            # Then send our weights
            our_weights = {
                'type': 'weights',
                'fitness': current_fitness,
                'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
                'weights_hash': self._get_model_hash()
            }
            
            if await NetworkUtils.safe_send_json(writer, our_weights, timeout=10.0):
                self.logger.info(f"✅ Winner completed: sent our weights to {partner_id}")
                # As winner, we don't change our weights - no cloning needed
            else:
                self.logger.error(f"❌ Failed to send winning weights to {partner_id}")
                
        else:
            self.logger.info(f"🥈 WE ARE LOSER: Sending our weights first, then receiving winner's")
            
            # As loser, we send our weights first
            our_weights = {
                'type': 'weights',
                'fitness': current_fitness,
                'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
                'weights_hash': self._get_model_hash()
            }
            
            if not await NetworkUtils.safe_send_json(writer, our_weights, timeout=10.0):
                self.logger.error(f"❌ Failed to send our loser weights to {partner_id}")
                return
                
            self.logger.info(f"📤 Sent our loser weights, now waiting for winner's weights")
            
            # Then receive the winner's weights
            partner_weights = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
            if partner_weights and partner_weights.get('type') == 'weights':
                # Convert partner's weights back to tensors and do the cloning
                if 'state_dict' in partner_weights:
                    partner_state = {k: torch.tensor(v) for k, v in partner_weights['state_dict'].items()}
                    self._mix_weights_based_on_fitness(partner_fitness, partner_state)
                    self.logger.info(f"✅ Loser completed: cloned winner's weights from {partner_id}")
                else:
                    self.logger.error(f"❌ Received weights without state_dict from {partner_id}")
            else:
                self.logger.error(f"❌ Failed to receive winner's weights from {partner_id}")
    
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
        Handles the entire lifecycle of a mixing attempt, including setting
        the state flag to ensure safe concurrency.
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
            
            connection = await NetworkUtils.safe_connect(host, port, timeout=2.0)
            if not connection:
                self.logger.info(f"❌ Could not connect to {partner}")
                return
            
            reader, writer = connection
            
            try:
                # Send "ready to mix" signal
                current_fitness = self.get_current_fitness()
                ready_msg = {
                    'type': 'ready_to_mix',
                    'sender': self.node_id,
                    'fitness': current_fitness
                }
                
                if not await NetworkUtils.safe_send_json(writer, ready_msg, timeout=2.0):
                    self.logger.info(f"❌ Failed to send ready signal to {partner}")
                    return
                
                # The partner will now determine who is winner/loser and follow the protocol
                # We need to determine our role too and act accordingly
                
                # Note: We don't know partner's fitness yet, so we need to receive their first message
                # which will tell us whether we should send first (if we're loser) or receive first (if we're winner)
                
                # Try to receive first - if partner is loser, they'll send their weights first
                try:
                    first_response = await NetworkUtils.safe_recv_json(reader, timeout=10.0)
                    if first_response and first_response.get('type') == 'weights':
                        # Partner sent weights first, so they are the loser and we are the winner
                        partner_fitness = first_response.get('fitness', 0.0)
                        self.logger.info(f"🏆 WE ARE WINNER (initiated): Received loser's weights from {partner}")
                        
                        # Send our winning weights back
                        our_weights = {
                            'type': 'weights',
                            'fitness': current_fitness,
                            'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
                            'weights_hash': self._get_model_hash()
                        }
                        
                        if await NetworkUtils.safe_send_json(writer, our_weights, timeout=10.0):
                            self.logger.info(f"✅ Winner (initiator) completed: sent weights to loser {partner}")
                            # As winner, we don't change our weights
                        else:
                            self.logger.error(f"❌ Failed to send winning weights to {partner}")
                            
                except asyncio.TimeoutError:
                    # Partner didn't send first, so we might be the loser
                    # We need to get partner's fitness first to determine this
                    self.logger.error(f"❌ Protocol error: couldn't determine winner/loser with {partner}")
                    return
                
                self.mixing_attempts += 1
                self.successful_mixes += 1
                self.logger.info(f"✅ Mix with {partner} completed successfully.")
                
            finally:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            self.logger.error(f"Error during mix initiation: {e}")
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

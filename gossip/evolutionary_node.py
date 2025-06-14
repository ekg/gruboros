import asyncio
import random
import torch
import io
import time
import os
import logging
from asyncio import Queue
from typing import Dict
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils
import uvloop

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module,
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_probability = mixing_probability
        
        self.mix_request_queue = Queue(maxsize=1)
        self.mixing_lock = asyncio.Lock()
        self.is_mixing = False  # Prevent concurrent mixing operations
        self.fitness_tracker = FitnessTracker()
        self.peer_list: Dict[str, dict] = {}
        
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        self.gossip_port = NetworkUtils.get_gossip_port(data_parallel_rank, master_addr)
        self.server = None
        self.gossip_running = False
        
        self.logger = logging.getLogger(f'evolutionary_node_{self.node_id}')
        
        self.bootstrap_nodes = NetworkUtils.get_bootstrap_nodes(
            global_rank, world_size, data_parallel_rank, tp_size, self.logger
        )
        
        self.mixing_attempts = 0
        self.successful_mixes = 0
        
    def update_fitness(self, loss_value: float):
        self.fitness_tracker.update(loss_value)
    
    def get_current_fitness(self) -> float:
        return self.fitness_tracker.get_fitness()

    def _perform_losing_clone(self, partner_weights_bytes: bytes):
        self.logger.info(f"🧬 Received {len(partner_weights_bytes)/1e6:.2f} MB payload. Applying new weights...")
        try:
            device = next(self.model.parameters()).device
            buffer = io.BytesIO(partner_weights_bytes)
            partner_state_dict = torch.load(buffer, map_location=device)
            self.model.load_state_dict(partner_state_dict)
            self.logger.info(f"  🔥 CLONING COMPLETE.")
            self.successful_mixes += 1
        except Exception as e:
            self.logger.error(f"  ❌ FAILED to clone partner weights: {e}")
    
    
    
    async def _handle_peer_connection(self, reader, writer):
        """Handle incoming connections with proper busy rejection"""
        # Check if already mixing
        if self.mixing_lock.locked():
            try:
                await NetworkUtils.send_message(writer, b"BUSY")
            except:
                pass
            writer.close()
            return
            
        try:
            await self._run_protocol(reader, writer, is_initiator=False)
        except Exception as e:
            self.logger.error(f"Protocol error: {e}")
        finally:
            if writer and not writer.is_closing():
                writer.close()
    
    
    async def _initiate_mix(self):
        """Entry point for outgoing mix attempts."""
        # Skip if already mixing to prevent race conditions
        if self.is_mixing:
            self.logger.debug("Skipping mix attempt - already mixing")
            return
            
        my_address = f"{os.environ.get('MASTER_ADDR', 'localhost')}:{self.gossip_port}"
        other_peers = [p for p in self.peer_list.keys() if p != my_address]
        if not other_peers:
            return

        self.mixing_attempts += 1
        partner_id = self.mixing_rng.choice(other_peers)
        
        connection = await NetworkUtils.safe_connect(partner_id)
        if not connection:
            self.logger.warning(f"❌ Could not connect to {partner_id}")
            return
        
        reader, writer = connection
        try:
            await self._run_protocol(reader, writer, is_initiator=True)
        except Exception as e:
            self.logger.error(f"Unhandled error in mix initiator: {e}")
        finally:
            if writer and not writer.is_closing():
                writer.close()

    async def _run_protocol(self, reader, writer, is_initiator):
        # Check if already mixing - prevent race conditions
        if self.is_mixing:
            if not is_initiator: 
                await NetworkUtils.send_message(writer, b"BUSY")
            return
            
        # Set mixing flag for entire operation
        self.is_mixing = True
        try:
            # Quick handshake phase
            if is_initiator:
                our_fitness = self.get_current_fitness()
                partner_addr = writer.get_extra_info('peername')
                self.logger.info(f"🎲 INITIATING MIXING with {partner_addr} (fitness: {our_fitness:.4f})")
                probe = f"PROBE|{self.node_id}|{our_fitness:.6f}"
                await NetworkUtils.send_message(writer, probe.encode())
                
                decision_bytes = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not decision_bytes: return
                
                decision = decision_bytes.decode()
                if decision == "BUSY": return
                we_are_winner = (decision == "YOU_WIN")
            else: # We are the peer
                probe_bytes = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not probe_bytes or not probe_bytes.decode().startswith("PROBE"): return
                
                _, partner_node_id, peer_fitness_str = probe_bytes.decode().split('|')
                peer_fitness = float(peer_fitness_str)
                our_fitness = self.get_current_fitness()
                self.logger.info(f"📬 Received probe from {partner_node_id} (fitness: {peer_fitness:.4f}). Our fitness: {our_fitness:.4f}")
                
                we_are_winner = (our_fitness > peer_fitness)
                response = b"I_WIN" if we_are_winner else b"YOU_WIN"
                await NetworkUtils.send_message(writer, response)
            
            # Weight transfer phase (protected by is_mixing flag)
            if we_are_winner:
                self.logger.info("🏆 We are the winner. Preparing and sending weights...")
                await self._send_weights(writer)
            else: # We are the loser
                self.logger.info("🥈 We are the loser. Waiting for weights...")
                await self._receive_weights(reader)
                
        finally:
            # Always clear mixing flag
            self.is_mixing = False

    async def _send_weights(self, writer):
        """Debug version with granular logging"""
        self.logger.info("🔧 _send_weights: Starting weight preparation")
        
        try:
            # Step 1: Serialize weights
            self.logger.info("🔧 _send_weights: Creating buffer")
            buffer = io.BytesIO()
            
            self.logger.info("🔧 _send_weights: Calling torch.save")
            torch.save(self.model.state_dict(), buffer)
            
            self.logger.info("🔧 _send_weights: Getting bytes from buffer")
            weights_bytes = buffer.getvalue()
            
            self.logger.info(f"🔧 _send_weights: Serialized {len(weights_bytes)/1e6:.2f} MB")
            
            # Step 2: Test connection is alive
            self.logger.info("🔧 _send_weights: Testing connection")
            if writer.is_closing():
                self.logger.error("🔧 _send_weights: Writer is already closing!")
                return
                
            # Step 3: Send full weights
            self.logger.info("🔧 _send_weights: Starting full weight transfer")
            await NetworkUtils.send_message(writer, weights_bytes)
            
            self.logger.info("✅ _send_weights: All operations completed successfully")
            self.successful_mixes += 1
            
        except asyncio.TimeoutError:
            self.logger.error("🔧 _send_weights: TIMEOUT during send operation")
        except Exception as e:
            self.logger.error(f"🔧 _send_weights: ERROR: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"🔧 _send_weights: Traceback: {traceback.format_exc()}")

    async def _receive_weights(self, reader):
        """Receives and applies weights."""
        weights_bytes = await NetworkUtils.receive_message(reader, timeout=300.0)
        if weights_bytes:
            self._perform_losing_clone(weights_bytes)
        else:
            self.logger.warning("Failed to receive weights (timed out or connection closed).")
    
    async def start_gossip_protocol(self):
        self.gossip_running = True
        try:
            # CRITICAL: Increase buffer limit to 500MB for large weight transfers
            # OPTIMIZATION: Add reuse_port=True, which is highly effective with uvloop.
            self.server = await asyncio.start_server(
                self._handle_peer_connection,
                '0.0.0.0',
                self.gossip_port,
                limit=500 * 1024 * 1024,  # 500MB buffer limit
                reuse_port=True
            )
            self.mixer_task = asyncio.create_task(self._mixer_loop())
            self.logger.info(f"Node {self.node_id}: Gossip protocol started on port {self.gossip_port} with uvloop backend.")
        except Exception as e:
            self.logger.error(f"Failed to start gossip protocol: {e}")
            self.gossip_running = False

    async def _mixer_loop(self):
        # Discover peers once at the start, then start mixing
        await self._discover_peers()
        while self.gossip_running:
            try:
                await self.mix_request_queue.get()
                await self._initiate_mix()
                self.mix_request_queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e:
                self.logger.error(f"Error in mixer loop: {e}")
                await asyncio.sleep(1)
    
    def request_mix(self):
        if random.random() < self.mixing_probability:
            try:
                self.mix_request_queue.put_nowait(True)
            except asyncio.QueueFull:
                pass
    
    async def _discover_peers(self):
        for bootstrap_addr in self.bootstrap_nodes:
            self.peer_list[bootstrap_addr] = {}

    async def stop_gossip_protocol(self):
        self.gossip_running = False
        if self.mixer_task:
            self.mixer_task.cancel()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.logger.info("Gossip protocol stopped.")
    
    def get_status(self) -> dict:
        return {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'recent_loss': self.fitness_tracker.get_recent_loss(),
            'peer_count': len(self.peer_list),
            'mixing_attempts': self.mixing_attempts,
            'successful_mixes': self.successful_mixes
        }

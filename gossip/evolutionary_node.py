import asyncio
import random
import torch
import io
import time
import os
import logging
from asyncio import Queue
from typing import Dict, List, Optional
from .fitness_tracker import FitnessTracker
from .network_utils import NetworkUtils

class EvolutionaryTrainingNode:
    def __init__(self, node_id: str, model: torch.nn.Module,
                 global_rank: int, world_size: int, data_parallel_rank: int,
                 tp_size: int, mixing_probability: float = 0.01):
        self.node_id = node_id
        self.model = model
        self.global_rank = global_rank
        # Use a per-process RNG for mixing decisions to prevent synchronization
        self.mixing_rng = random.Random(42 + global_rank * 1000)
        self.mixing_probability = mixing_probability
        
        self.mix_request_queue = Queue(maxsize=1)
        self.mixing_lock = asyncio.Lock()  # Prevents deadlock
        self.fitness_tracker = FitnessTracker()
        self.peer_list: Dict[str, dict] = {}
        
        # Network setup
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
        self.logger.info(f"🧬 EVOLUTIONARY CLONING: Loading new weights...")
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
        async with self.mixing_lock:
            await self._run_gossip_protocol(reader, writer, is_initiator=False)
    
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
        partner_addr = writer.get_extra_info('peername')
        try:
            if is_initiator:
                # 1. INITIATOR: Sends probe
                partner_id = self.mixing_rng.choice(list(self.peer_list.keys()))
                current_fitness = self.get_current_fitness()
                self.logger.info(f"🎲 INITIATING MIXING with {partner_id} (fitness: {current_fitness:.4f})")
                probe = f"PROBE|{self.node_id}|{current_fitness:.6f}"
                await NetworkUtils.send_message(writer, probe.encode())

                # 2. INITIATOR: Waits for peer's decision
                response_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not response_data:
                    self.logger.warning(f"No response from {partner_id}.")
                    return

                response = response_data.decode()
                if response == "RESPONSE|LOSER":
                    # 3a. INITIATOR (we won): The peer is now waiting. We prepare and send the payload.
                    self.logger.info(f"🏆 Peer {partner_id} is loser. Preparing to send weights...")
                    
                    buffer = io.BytesIO()
                    torch.save(self.model.state_dict(), buffer)
                    weights_bytes = buffer.getvalue()
                    
                    self.logger.info(f"Sending weights to {partner_id} ({len(weights_bytes)/1e6:.2f} MB).")
                    await NetworkUtils.send_message(writer, weights_bytes)
                    self.logger.info(f"✅ Sent weights successfully.")
                    self.successful_mixes += 1
                else: # Response is "WINNER"
                    # 3b. INITIATOR (we lost): The peer has closed the connection. The mix is over for us.
                    self.logger.info(f"🥈 Peer {partner_id} is winner. Ending mix.")

            else:  # I am the PEER
                # 1. PEER: Receives probe
                probe_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not probe_data: return
                
                probe = probe_data.decode()
                if not probe.startswith("PROBE"): return
                
                _, partner_node_id, peer_fitness_str = probe.split('|')
                peer_fitness = float(peer_fitness_str)
                current_fitness = self.get_current_fitness()
                self.logger.info(f"📬 Received probe from {partner_node_id} (fitness: {peer_fitness:.4f}). Our fitness: {current_fitness:.4f}")

                # 2. PEER: Makes decision
                if peer_fitness > current_fitness:
                    # 3a. PEER (we are the loser): We tell the initiator it won, then wait patiently for the payload.
                    self.logger.info(f"🥈 We are the loser. Sending LOSER response and waiting for weights...")
                    await NetworkUtils.send_message(writer, b"RESPONSE|LOSER")
                    
                    # [THE CRITICAL FIX] Increase the timeout significantly here.
                    # This is the ONLY place a long timeout is needed.
                    # 5 minutes should be more than enough for any model serialization.
                    weights_bytes = await NetworkUtils.receive_message(reader, timeout=300.0)
                    
                    if weights_bytes:
                        self.logger.info(f"Received {len(weights_bytes)/1e6:.2f} MB of weights from winner {partner_node_id}.")
                        self._perform_losing_clone(weights_bytes)
                    else:
                        self.logger.warning(f"Connection timed out or was closed while waiting for weights from {partner_node_id}.")
                else:
                    # 3b. PEER (we are the winner): We tell the initiator it lost. The mix is over for us.
                    self.logger.info(f"🏆 We are the winner. Sending WINNER response and ending mix.")
                    await NetworkUtils.send_message(writer, b"RESPONSE|WINNER")

        except asyncio.TimeoutError:
            self.logger.warning(f"⏳ Connection with {partner_addr} timed out during initial handshake.")
        except (ConnectionResetError, BrokenPipeError):
             self.logger.warning(f"Connection with {partner_addr} was closed unexpectedly.")
        except Exception as e:
            self.logger.error(f"Error during gossip with {partner_addr}: {type(e).__name__} - {e}")
    
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
                await asyncio.sleep(self.mixing_rng.uniform(10, 30))
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
        if self.mixing_rng.random() < self.mixing_probability:
            try:
                self.mix_request_queue.put_nowait(True)
            except asyncio.QueueFull:
                pass
    
    async def _initiate_mix(self):
        if not self.peer_list: return
        self.mixing_attempts += 1
        
        async with self.mixing_lock:
            partner_id = self.mixing_rng.choice(list(self.peer_list.keys()))
            host, port_str = partner_id.split(':')
            port = int(port_str)
            
            connection = await NetworkUtils.safe_connect(host, port, timeout=5.0)
            if not connection:
                self.logger.warning(f"❌ Could not connect to {partner_id}")
                return
            
            reader, writer = connection
            try:
                await self._run_gossip_protocol(reader, writer, is_initiator=True)
            finally:
                if writer and not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
    
    async def _discover_peers(self):
        for bootstrap_addr in self.bootstrap_nodes:
            if bootstrap_addr not in self.peer_list:
                self.peer_list[bootstrap_addr] = {}
    
    async def stop_gossip_protocol(self):
        """Gracefully shuts down the gossip protocol components."""
        self.logger.info("Stopping gossip protocol...")
        
        # 1. Signal all background tasks to stop their loops
        self.gossip_running = False

        # 2. Cancel the mixer task so no new mixes are initiated.
        if hasattr(self, 'mixer_task') and self.mixer_task:
            self.mixer_task.cancel()
            # Wait for the task to acknowledge cancellation
            try:
                await self.mixer_task
            except asyncio.CancelledError:
                self.logger.info("Mixer task successfully cancelled.")

        # 3. Cancel other background tasks
        if self.gossip_task:
            self.gossip_task.cancel()
            try:
                await self.gossip_task
            except asyncio.CancelledError:
                self.logger.info("Gossip task successfully cancelled.")

        # 4. Stop accepting new connections
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Gossip server closed.")

        # 5. Wait briefly for any final in-flight operation protected by the lock
        try:
            await asyncio.wait_for(self.mixing_lock.acquire(), timeout=2.0)
            self.mixing_lock.release()
        except asyncio.TimeoutError:
            self.logger.warning("Timed out waiting for final mix to complete.")
        
        success_rate = (self.successful_mixes / max(1, self.mixing_attempts)) * 100
        self.logger.info(f"Gossip stopped. Success rate: {success_rate:.1f}%")
    
    def get_status(self) -> dict:
        return {
            'node_id': self.node_id,
            'fitness': self.get_current_fitness(),
            'recent_loss': self.fitness_tracker.get_recent_loss(),
            'peer_count': len(self.peer_list),
            'mixing_attempts': self.mixing_attempts,
            'successful_mixes': self.successful_mixes
        }

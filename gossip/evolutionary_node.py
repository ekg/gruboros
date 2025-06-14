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
        # [ADD THIS LOG]
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
        # [CRITICAL FIX] Implement the "BUSY" check to prevent head-of-line blocking.
        # First, check if we are busy without blocking.
        if self.mixing_lock.locked():
            peer_addr = writer.get_extra_info('peername')
            self.logger.info(f"Received mix request from {peer_addr} while busy. Responding with BUSY.")
            try:
                # Tell the initiator we're busy and it should try again later.
                await NetworkUtils.send_message(writer, b"RESPONSE|BUSY")
            except (ConnectionResetError, BrokenPipeError):
                pass # The initiator might have already timed out and closed.
            finally:
                writer.close()
                await writer.wait_closed()
            return # End this handler immediately.

        # If we are not busy, acquire the lock and handle the full protocol.
        async with self.mixing_lock:
            await self._run_gossip_protocol(reader, writer, is_initiator=False)
    
    
    async def _run_gossip_protocol(self, reader, writer, is_initiator: bool):
        """
        [THE DEFINITIVE FIX] A robust, symmetrical protocol.
        The winner ALWAYS sends weights. The loser ALWAYS receives.
        """
        partner_addr = writer.get_extra_info('peername')
        try:
            if is_initiator:
                partner_id = self.mixing_rng.choice(list(self.peer_list.keys()))
                our_fitness = self.get_current_fitness()
                self.logger.info(f"🎲 INITIATING MIXING with {partner_id} (fitness: {our_fitness:.4f})")
                probe = f"PROBE|{self.node_id}|{our_fitness:.6f}"
                await NetworkUtils.send_message(writer, probe.encode())

                # Step 2: Initiator waits for the peer's simple decision.
                response_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not response_data:
                    self.logger.warning(f"No response from {partner_id}.")
                    return

                response = response_data.decode()
                if response == "RESPONSE|BUSY":
                    self.logger.info(f"Peer {partner_id} is busy. Will try again later.")
                    return
                
                # Step 3: Act based on the decision.
                if response == "RESPONSE|WINNER": # This means WE LOST
                    self.logger.info(f"🥈 We are the loser. Waiting for weights from {partner_id}.")
                    weights_bytes = await NetworkUtils.receive_message(reader, timeout=300.0)
                    if weights_bytes:
                        self.logger.info(f"Received {len(weights_bytes)/1e6:.2f} MB of weights from winner {partner_id}.")
                        self._perform_losing_clone(weights_bytes)
                    else:
                        self.logger.warning(f"Connection timed out or was closed while waiting for weights from {partner_id}.")

                elif response == "RESPONSE|LOSER": # This means WE WON
                    self.logger.info(f"🏆 We are the winner. Preparing to send weights to {partner_id}.")
                    buffer = io.BytesIO()
                    torch.save(self.model.state_dict(), buffer)
                    weights_bytes = buffer.getvalue()
                    
                    self.logger.info(f"Sending weights ({len(weights_bytes)/1e6:.2f} MB).")
                    await NetworkUtils.send_message(writer, weights_bytes)
                    self.logger.info(f"✅ Sent weights successfully.")
                    self.successful_mixes += 1

            else:  # I am the PEER
                probe_data = await NetworkUtils.receive_message(reader, timeout=15.0)
                if not probe_data: return
                
                probe = probe_data.decode()
                if not probe.startswith("PROBE"): return
                
                _, partner_node_id, peer_fitness = probe.split('|')
                peer_fitness = float(peer_fitness)
                our_fitness = self.get_current_fitness()
                self.logger.info(f"📬 Received probe from {partner_node_id} (fitness: {peer_fitness:.4f}). Our fitness: {our_fitness:.4f}")

                # Step 2: Peer makes decision and acts.
                if our_fitness > peer_fitness: # WE ARE THE WINNER
                    self.logger.info(f"🏆 We are the winner. Sending WINNER response and preparing weights.")
                    await NetworkUtils.send_message(writer, b"RESPONSE|WINNER")
                    
                    buffer = io.BytesIO()
                    torch.save(self.model.state_dict(), buffer)
                    weights_bytes = buffer.getvalue()
                    
                    self.logger.info(f"Sending weights ({len(weights_bytes)/1e6:.2f} MB).")
                    await NetworkUtils.send_message(writer, weights_bytes)
                    self.logger.info(f"✅ Sent weights successfully.")
                    self.successful_mixes += 1

                else: # WE ARE THE LOSER
                    self.logger.info(f"🥈 We are the loser. Sending LOSER response and waiting for weights.")
                    await NetworkUtils.send_message(writer, b"RESPONSE|LOSER")
                    
                    weights_bytes = await NetworkUtils.receive_message(reader, timeout=300.0)
                    if weights_bytes:
                        self.logger.info(f"Received {len(weights_bytes)/1e6:.2f} MB of weights from winner {partner_node_id}.")
                        self._perform_losing_clone(weights_bytes)
                    else:
                        self.logger.warning(f"Connection timed out or was closed while waiting for weights from {partner_node_id}.")
        except asyncio.TimeoutError:
            self.logger.warning(f"⏳ Connection with {partner_addr} timed out during initial handshake.")
        except (ConnectionResetError, BrokenPipeError):
             self.logger.warning(f"Connection with {partner_addr} was closed unexpectedly.")
        except Exception as e:
            self.logger.error(f"Error during gossip with {partner_addr}: {type(e).__name__} - {e}")
    
    async def start_gossip_protocol(self):
        self.gossip_running = True
        try:
            self.server = await asyncio.start_server(
                self._handle_peer_connection, '0.0.0.0', self.gossip_port
            )
            # Create background tasks
            self.gossip_task = asyncio.create_task(self._gossip_loop())
            self.mixer_task = asyncio.create_task(self._mixer())
            self.logger.info(f"Node {self.node_id}: Gossip protocol started on port {self.gossip_port}")
        except Exception as e:
            self.logger.error(f"Failed to start gossip protocol: {e}")
            self.gossip_running = False
    
    async def _gossip_loop(self):
        while self.gossip_running:
            try:
                await self._discover_peers()
                await asyncio.sleep(self.mixing_rng.uniform(10, 30))
            except asyncio.CancelledError: break
            except Exception as e:
                self.logger.error(f"Gossip loop error: {e}")
                await asyncio.sleep(30)
    
    async def _mixer(self):
        while self.gossip_running:
            try:
                await self.mix_request_queue.get()
                await self._initiate_mix()
                self.mix_request_queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e:
                self.logger.error(f"Error in mixer task: {e}")
                await asyncio.sleep(1)
    
    def request_mix(self):
        # NOTE: This uses the global `random` module. It's CRITICAL that
        # `train.py` re-seeds it with the global_rank after deepspeed.initialize()
        if random.random() < self.mixing_probability:
            try:
                self.mix_request_queue.put_nowait(True)
            except asyncio.QueueFull:
                pass # A mix is already requested, that's fine.
    
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
        self.gossip_running = False
        for task in [self.mixer_task, self.gossip_task]:
            if task:
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
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

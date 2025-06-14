import asyncio
import json
import socket
import logging
import os
import glob
import subprocess
import torch
import torch.distributed as dist
from typing import List, Optional, Tuple

class NetworkUtils:
    @staticmethod
    def get_bootstrap_nodes(global_rank: int, world_size: int, data_parallel_rank: int, 
                          tp_size: int, logger) -> List[str]:
        """
        Smart peer discovery that adapts to deployment:
        - Local multi-GPU: Different ports per rank (multiple models per node)
        - Frontier: Same port per node (one model per node via tensor parallelism)
        """
        bootstrap_addresses = []
        base_gossip_port = 29501
        
        # Get master address from DeepSpeed
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        
        # Determine deployment mode
        is_local_multirank = (master_addr in ['localhost', '127.0.0.1'])
        data_parallel_world_size = world_size // tp_size
        
        if is_local_multirank:
            # LOCAL MODE: Multiple data parallel replicas on same machine
            # Each replica needs different port
            logger.info(f"Local deployment detected: {data_parallel_world_size} models on {master_addr}")
            
            for dp_rank in range(data_parallel_world_size):
                if dp_rank != data_parallel_rank:  # Don't add self
                    peer_port = base_gossip_port + dp_rank
                    bootstrap_addresses.append(f"{master_addr}:{peer_port}")
                    
        else:
            # DISTRIBUTED MODE: One data parallel replica per node
            # All nodes can use same port since they're on different machines
            logger.info(f"Distributed deployment detected: {data_parallel_world_size} nodes")
            
            # We need to discover other node addresses
            # Try to get them from SLURM or hostfile
            peer_nodes = NetworkUtils._discover_peer_nodes(logger)
            
            if peer_nodes:
                # Use discovered nodes
                for node in peer_nodes:
                    if node != master_addr:  # Don't add self
                        bootstrap_addresses.append(f"{node}:{base_gossip_port}")
            else:
                # Fallback: assume sequential node naming (common pattern)
                # This is a heuristic for when we can't discover nodes
                logger.warning("Could not discover peer nodes, using heuristic naming")
                
                # Extract numeric part from master_addr if possible
                import re
                match = re.search(r'(\d+)$', master_addr)
                if match:
                    base_num = int(match.group(1))
                    base_name = master_addr[:match.start()]
                    
                    # Generate peer node names
                    for i in range(data_parallel_world_size):
                        if i != data_parallel_rank:
                            peer_node = f"{base_name}{base_num + i}"
                            bootstrap_addresses.append(f"{peer_node}:{base_gossip_port}")
        
        logger.info(f"Rank {global_rank} (DP rank {data_parallel_rank}): "
                   f"Bootstrap peers: {bootstrap_addresses}")
        
        return bootstrap_addresses
    
    @staticmethod
    def _discover_peer_nodes(logger) -> List[str]:
        """Discover peer node hostnames from SLURM or hostfile"""
        nodes = []
        
        # Method 1: Use existing DeepSpeed hostfile  
        slurm_job_id = os.environ.get('SLURM_JOB_ID', '*')
        hostfile_pattern = f"hostfile-job{slurm_job_id}.txt"
        hostfiles = glob.glob(hostfile_pattern)
        
        if hostfiles:
            try:
                with open(hostfiles[0], 'r') as f:
                    for line in f:
                        if 'slots=' in line:
                            hostname = line.split()[0]
                            if hostname not in nodes:
                                nodes.append(hostname)
                logger.info(f"Found {len(nodes)} nodes from hostfile: {hostfiles[0]}")
                return nodes
            except Exception as e:
                logger.warning(f"Failed to read hostfile: {e}")
        
        # Method 2: SLURM environment
        if 'SLURM_JOB_NODELIST' in os.environ:
            try:
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], 
                    capture_output=True, text=True
                )
                nodes = result.stdout.strip().split('\n')
                logger.info(f"Found {len(nodes)} nodes from SLURM_JOB_NODELIST")
                return nodes
            except Exception as e:
                logger.warning(f"Failed to parse SLURM_JOB_NODELIST: {e}")
        
        return []
    
    @staticmethod
    def get_gossip_port(data_parallel_rank: int, master_addr: str) -> int:
        """Get the gossip port for this data parallel replica"""
        base_port = 29501
        
        # Local deployment: offset by data parallel rank
        if master_addr in ['localhost', '127.0.0.1']:
            return base_port + data_parallel_rank
        else:
            # Distributed deployment: same port on all nodes
            return base_port
    
    @staticmethod
    async def safe_connect(host: str, port: int, timeout: float = 5.0) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Safely connect to a peer with timeout"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            return reader, writer
        except Exception:
            return None
    
    @staticmethod
    async def send_message(writer: asyncio.StreamWriter, message: bytes):
        """
        [THE CORRECT ASYNCIO FIX] Send a large message without deadlocking the event loop.
        """
        try:
            # Prepare the full payload with a length prefix.
            len_bytes = len(message).to_bytes(4, 'big')
            
            # Write the length prefix and the message to the stream's buffer.
            # These are non-blocking operations.
            writer.write(len_bytes)
            writer.write(message)
            
            # The KEY to avoiding the deadlock:
            # Instead of `await writer.drain()`, which can block the event loop,
            # we simply yield control back to the loop for one cycle.
            # This gives the loop an opportunity to process the I/O and actually
            # start sending the data from the buffer over the network.
            # The final `writer.close()` will handle the draining implicitly.
            await asyncio.sleep(0)

        except (ConnectionResetError, BrokenPipeError):
            logging.warning("send_message failed: Connection was closed by peer.")
        except Exception as e:
            logging.error(f"send_message failed with unexpected error: {e}")

    @staticmethod
    async def receive_message(reader: asyncio.StreamReader, timeout: float = 5.0) -> Optional[bytes]:
        try:
            # First, read the 4-byte length prefix with a short timeout.
            len_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
            msg_len = int.from_bytes(len_bytes, 'big')

            # Now, read the full payload with a much longer timeout,
            # giving the sender time to prepare and transmit the large model.
            if msg_len > 500 * 1024 * 1024: # Safety check for > 500MB
                 logging.error(f"Message size {msg_len/1e6:.2f} MB exceeds limit.")
                 return None
                 
            # Use a long, fixed timeout for receiving the actual large payload.
            payload_timeout = 300.0
            return await asyncio.wait_for(reader.readexactly(msg_len), timeout=payload_timeout)
        except asyncio.IncompleteReadError:
            logging.warning("receive_message failed: Incomplete read, peer likely closed connection.")
            return None
        except asyncio.TimeoutError:
            # This will now correctly trigger if the payload isn't received in time.
            logging.warning(f"receive_message timed out after waiting {timeout}s (for header) or 300s (for payload).")
            return None
        except (ConnectionResetError, BrokenPipeError):
            logging.warning("receive_message failed: Connection was closed by peer.")
            return None

    @staticmethod
    async def safe_send_json(writer: asyncio.StreamWriter, data: dict, timeout: float = 5.0) -> bool:
        """Safely send JSON data"""
        try:
            json_data = json.dumps(data).encode()
            length = len(json_data)
            # Send length first, then data
            writer.write(length.to_bytes(4, byteorder='big'))
            writer.write(json_data)
            await asyncio.wait_for(writer.drain(), timeout=timeout)
            return True
        except Exception:
            return False
    
    @staticmethod
    async def safe_recv_json(reader: asyncio.StreamReader, timeout: float = 5.0) -> Optional[dict]:
        """Safely receive JSON data"""
        try:
            # Read length first
            length_bytes = await asyncio.wait_for(reader.read(4), timeout=timeout)
            if len(length_bytes) != 4:
                return None
            
            length = int.from_bytes(length_bytes, byteorder='big')
            if length > 10 * 1024 * 1024:  # 10MB limit
                return None
            
            # Read actual data
            data = await asyncio.wait_for(reader.read(length), timeout=timeout)
            return json.loads(data.decode())
        except Exception:
            return None

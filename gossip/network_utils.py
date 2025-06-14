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
    async def safe_connect(partner_id: str) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        try:
            host, port_str = partner_id.split(':')
            port = int(port_str)
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5.0)
        except Exception:
            return None
    
    @staticmethod
    async def send_message(writer: asyncio.StreamWriter, data: bytes):
        """Send binary data with length prefix (proven pattern)"""
        try:
            # Send 4-byte length prefix (network byte order)
            size_bytes = len(data).to_bytes(4, byteorder='big')
            writer.write(size_bytes)
            writer.write(data)
            # Critical: ensure all data is sent before returning
            await writer.drain()
        except Exception as e:
            logging.error(f"send_message failed: {e}")
            raise

    @staticmethod
    async def receive_message(reader: asyncio.StreamReader, timeout: float = 60.0) -> Optional[bytes]:
        """Receive binary data with length prefix (proven pattern)"""
        try:
            # Read exactly 4 bytes for the length prefix
            size_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
            size = int.from_bytes(size_bytes, byteorder='big')
            
            # Sanity check
            if size > 500 * 1024 * 1024:  # 500MB limit
                logging.error(f"Message size {size/1e6:.2f} MB exceeds limit")
                return None
            
            # Read exactly the number of bytes specified
            data = await asyncio.wait_for(reader.readexactly(size), timeout=timeout)
            return data
            
        except asyncio.IncompleteReadError as e:
            logging.warning(f"Incomplete read: got {len(e.partial)} bytes, expected more")
            return None
        except asyncio.TimeoutError:
            logging.warning(f"Receive timeout after {timeout}s")
            return None
        except Exception as e:
            logging.error(f"receive_message failed: {e}")
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

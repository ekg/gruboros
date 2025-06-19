import socket
import json
import logging
import os
import glob
import subprocess
import torch
import torch.distributed as dist
import time
from typing import List, Optional, Tuple

class NetworkUtils:
    @staticmethod
    def optimize_socket(sock):
        """Apply high-performance TCP settings"""
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
        if hasattr(socket, 'TCP_QUICKACK'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
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
    def get_local_node_address() -> str:
        """Get THIS node's actual IP address, not the master's"""
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        
        # Local multi-GPU case
        if master_addr in ['localhost', '127.0.0.1']:
            return 'localhost'
        
        # Distributed case - get our actual IP
        try:
            # Connect to master to determine which interface we'd use
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((master_addr, 80))
                local_ip = s.getsockname()[0]
                return local_ip
        except Exception:
            # Fallback for HPC systems
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)

    @staticmethod
    def get_node_identity(data_parallel_rank: int) -> tuple:
        """Get (ip_address, port) for this data parallel replica"""
        local_ip = NetworkUtils.get_local_node_address()
        base_port = 29501
        
        # Local: different ports per DP rank
        # Distributed: same port (different IPs)
        if local_ip == 'localhost':
            port = base_port + data_parallel_rank
        else:
            port = base_port
            
        return local_ip, port

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
    

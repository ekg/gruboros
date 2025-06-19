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
    def get_gossip_port(local_rank: int) -> int:
        """
        Get a unique gossip port for this process on this node.
        The port is offset by the local_rank to avoid conflicts when multiple
        processes run on the same machine. This is robust for both local
        multi-GPU and multi-node distributed setups.
        """
        base_port = 29501  # Base port for gossip protocol
        # Each process on a node gets a unique port: base + local_rank
        return base_port + local_rank

    @staticmethod
    def get_bootstrap_nodes(global_rank: int, local_rank: int, world_size: int, data_parallel_rank: int, 
                            tp_size: int, logger) -> List[str]:
        """
        Smart peer discovery. It constructs peer addresses by finding the peer's
        hostname and assuming it's listening on a port offset by its local_rank.
        
        This assumes symmetric local_rank assignment across nodes.
        """
        bootstrap_addresses = []
        
        # Discover all potential peer node hostnames
        peer_nodes = NetworkUtils._discover_peer_nodes(logger)
        
        if not peer_nodes:
            logger.warning("Could not discover peer hostnames via hostfile or SLURM. "
                           "Gossip protocol may fail to find peers.")
            return []

        # The port for our peer will be the base port + our own local_rank.
        # This assumes that peer processes (e.g., all local_rank 0 processes)
        # form a communicating group.
        peer_port = NetworkUtils.get_gossip_port(local_rank)
        
        # Get our own hostname to avoid adding ourselves to the peer list
        my_hostname = socket.gethostname()

        for node_hostname in peer_nodes:
            # scontrol can sometimes return the FQDN, so we compare startswith
            if not node_hostname.startswith(my_hostname):
                bootstrap_addresses.append(f"{node_hostname}:{peer_port}")

        logger.info(f"Rank {global_rank} (local_rank {local_rank}): "
                    f"Bootstrap peers for port {peer_port}: {bootstrap_addresses}")
        
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
    def get_node_identity(local_rank: int) -> tuple:
        """Get (ip_address, port) for this specific process"""
        local_ip = NetworkUtils.get_local_node_address()
        port = NetworkUtils.get_gossip_port(local_rank)
        return local_ip, port
    

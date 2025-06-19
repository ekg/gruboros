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
    def get_bootstrap_nodes(global_rank: int, local_rank: int, logger) -> List[str]:
        """
        Discovers ALL other processes in the job to create a complete peer list
        for true all-to-all evolutionary mixing.
        """
        # Get my own hostname to identify myself in the list
        # Use socket.getfqdn() for fully qualified name to be safe
        try:
            my_hostname = socket.getfqdn()
        except:
            my_hostname = socket.gethostname()

        # Discover all node hostnames in the job from Slurm or hostfile
        all_nodes = NetworkUtils._discover_peer_nodes(logger)
        if not all_nodes:
            logger.error(f"FATAL: Rank {global_rank} could not discover any peer nodes. Gossip will fail.")
            return []

        # Get the number of ranks per node from the environment variable we set
        try:
            ranks_per_node = int(os.environ['RANKS_PER_NODE'])
        except (KeyError, ValueError):
            logger.warning("RANKS_PER_NODE env var not set or invalid. Falling back to 8 for Frontier.")
            ranks_per_node = 8

        bootstrap_addresses = []

        # Iterate through every node and every possible local_rank on that node
        for node_hostname in all_nodes:
            for peer_local_rank in range(ranks_per_node):
                # Check if this potential peer is me
                # Use startswith for safety, as scontrol can return short names
                is_me = node_hostname.startswith(my_hostname.split('.')[0]) and peer_local_rank == local_rank

                if not is_me:
                    # It's a peer. Calculate its port and add it to our list.
                    peer_port = NetworkUtils.get_gossip_port(peer_local_rank)
                    bootstrap_addresses.append(f"{node_hostname}:{peer_port}")

        logger.info(f"Rank {global_rank} (local_rank {local_rank}) discovered {len(bootstrap_addresses)} total peers.")
        if len(bootstrap_addresses) > 0 and len(bootstrap_addresses) < 10:
             logger.info(f"Peer sample: {bootstrap_addresses[:10]}")
             
        return bootstrap_addresses
    
    @staticmethod
    def _discover_peer_nodes(logger) -> List[str]:
        """Discover peer node hostnames from SLURM or hostfile"""
        nodes = []
        
        # Method 1: SLURM environment (most reliable on HPC)
        if 'SLURM_JOB_NODELIST' in os.environ:
            try:
                # Use scontrol to expand the nodelist into individual hostnames
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], 
                    capture_output=True, text=True, check=True
                )
                # Use a set to handle potential duplicates then convert to list
                nodes = sorted(list(set(result.stdout.strip().split('\n'))))
                logger.info(f"Discovered {len(nodes)} nodes from SLURM_JOB_NODELIST.")
                return nodes
            except Exception as e:
                logger.warning(f"Failed to parse SLURM_JOB_NODELIST via scontrol: {e}")

        # Method 2: Fallback to DeepSpeed hostfile
        slurm_job_id = os.environ.get('SLURM_JOB_ID', '*')
        hostfile_pattern = f"hostfile-job{slurm_job_id}.txt"
        hostfiles = glob.glob(hostfile_pattern)
        
        if hostfiles:
            try:
                with open(hostfiles[0], 'r') as f:
                    # Use a set to handle potential duplicates
                    node_set = set()
                    for line in f:
                        # Handle lines like "hostname slots=8"
                        if line.strip():
                            hostname = line.split()[0]
                            node_set.add(hostname)
                nodes = sorted(list(node_set))
                logger.info(f"Discovered {len(nodes)} nodes from hostfile: {hostfiles[0]}")
                return nodes
            except Exception as e:
                logger.warning(f"Failed to read hostfile '{hostfiles[0]}': {e}")
        
        logger.error("Could not discover peer nodes from SLURM or hostfile.")
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
    

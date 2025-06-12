import asyncio
import json
import socket
import logging
import os
import glob
import subprocess
from typing import List, Optional, Tuple

class NetworkUtils:
    @staticmethod
    def get_bootstrap_nodes(global_rank: int, logger) -> List[str]:
        """Get bootstrap nodes from existing SLURM hostfile or local setup"""
        nodes = []
        
        # Method 1: Use existing DeepSpeed hostfile (SLURM case)
        slurm_job_id = os.environ.get('SLURM_JOB_ID', '*')
        hostfile_pattern = f"hostfile-job{slurm_job_id}.txt"
        hostfiles = glob.glob(hostfile_pattern)
        
        if hostfiles:
            # Parse existing hostfile format: "hostname slots=8"
            with open(hostfiles[0], 'r') as f:
                for line in f:
                    if 'slots=' in line:
                        hostname = line.split()[0]
                        nodes.append(hostname)
            
            # Use first 5 nodes as bootstrap
            nodes = nodes[:5]
            logger.info(f"Found {len(nodes)} nodes from hostfile: {hostfiles[0]}")
        
        # Method 2: SLURM environment fallback
        elif 'SLURM_JOB_NODELIST' in os.environ:
            try:
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], 
                    capture_output=True, text=True
                )
                nodes = result.stdout.strip().split('\n')[:5]
                logger.info(f"Found {len(nodes)} nodes from SLURM_JOB_NODELIST")
            except Exception as e:
                logger.warning(f"Failed to parse SLURM_JOB_NODELIST: {e}")
        
        # Method 3: Local multi-GPU testing
        else:
            # For local testing, create fake hostfile
            if not os.path.exists("hostfile-job-local.txt"):
                with open("hostfile-job-local.txt", "w") as f:
                    f.write("localhost slots=8\n")
            nodes = ['localhost']
            logger.info("Using localhost for local multi-GPU testing")
        
        # Convert to address:port format (avoid conflict with MASTER_PORT)
        gossip_base_port = 29501
        return [f"{node}:{gossip_base_port + (global_rank % 100)}" for node in nodes]
    
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

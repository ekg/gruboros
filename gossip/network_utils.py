import asyncio
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
    async def send_message(writer: asyncio.StreamWriter, data: bytes, logger=None):
        """High-performance message sending with optimized TCP settings"""
        if logger is None:
            logger = logging.getLogger(__name__)  # Fallback
        try:
            # Get the underlying socket for optimization
            sock = writer.get_extra_info('socket')
            if sock:
                # Enable TCP_NODELAY for low latency (disable Nagle's algorithm)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set large socket buffers for high throughput
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
                
                # Enable TCP_QUICKACK on Linux for faster ACKs
                if hasattr(socket, 'TCP_QUICKACK'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            
            # Get the transport for buffer limit optimization
            transport = writer.transport
            if hasattr(transport, 'set_write_buffer_limits'):
                # Set very high write buffer limits to avoid premature blocking
                # This reduces the frequency of drain() calls
                transport.set_write_buffer_limits(high=64*1024*1024, low=32*1024*1024)  # 64MB/32MB
            
            # Send length prefix
            prefix = len(data).to_bytes(4, "big")
            writer.write(prefix)
            
            # ✅ Always log start with timing
            start_time = time.time()
            logger.info(f"🚀 Starting transfer: {len(data)/1e6:.2f} MB")
            
            # Send data with MINIMAL progress logging
            chunk_size = 8 * 1024 * 1024  # 8MB chunks for good performance
            bytes_sent = 0
            next_progress_threshold = 100 * 1024 * 1024  # Only log every 100MB
            drain_interval = 128 * 1024 * 1024  # Only drain every 128MB
            last_drain = 0
            
            while bytes_sent < len(data):
                chunk_end = min(bytes_sent + chunk_size, len(data))
                chunk = data[bytes_sent:chunk_end]
                
                # Write chunk without immediate drain
                writer.write(chunk)
                bytes_sent = chunk_end
                
                # ✅ Only log progress for LARGE transfers (>100MB) and infrequently
                if len(data) > 100 * 1024 * 1024 and bytes_sent >= next_progress_threshold:
                    logger.info(f"📤 Progress: {bytes_sent/1e6:.0f}/{len(data)/1e6:.0f} MB")
                    next_progress_threshold += 100 * 1024 * 1024  # Next 100MB
                
                # Only drain periodically or at the end
                if (bytes_sent - last_drain) >= drain_interval or bytes_sent == len(data):
                    await writer.drain()
                    last_drain = bytes_sent
                
                # Brief yield to event loop without blocking
                if bytes_sent % (64 * 1024 * 1024) == 0:  # Every 64MB
                    await asyncio.sleep(0)
            
            # ✅ Always log completion with speed
            elapsed = time.time() - start_time
            speed_mbps = (len(data) / 1e6) / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Transfer complete: {len(data)/1e6:.2f} MB in {elapsed:.2f}s ({speed_mbps:.0f} MB/s)")
            
        except ConnectionResetError:
            # ✅ Handle peer busy gracefully - don't log as error
            logger.warning("⚠️  Peer connection reset (likely busy) - transfer cancelled")
            raise ConnectionResetError("Peer busy")  # Re-raise but as expected condition
        except Exception as e:
            logger.error(f"❌ High-speed transfer failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    async def receive_message(reader: asyncio.StreamReader, timeout: float = 300.0, logger=None) -> Optional[bytes]:
        """High-performance message receiving with optimized settings"""
        if logger is None:
            logger = logging.getLogger(__name__)  # Fallback
        try:
            # Read length prefix
            prefix_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=30.0)
            expected_size = int.from_bytes(prefix_bytes, "big")
            
            # ✅ Always log start with timing
            start_time = time.time()
            logger.info(f"🔄 Receiving {expected_size/1e6:.2f} MB...")
            
            # Use large read chunks for better performance
            received_data = bytearray()
            bytes_remaining = expected_size
            read_chunk_size = 8 * 1024 * 1024  # 8MB read chunks
            next_progress_threshold = 100 * 1024 * 1024  # Only log every 100MB
            
            while bytes_remaining > 0:
                read_size = min(read_chunk_size, bytes_remaining)
                
                # Use readexactly for reliable large transfers
                chunk = await asyncio.wait_for(reader.readexactly(read_size), timeout=60.0)
                received_data.extend(chunk)
                bytes_remaining -= len(chunk)
                
                # ✅ Only log progress for LARGE transfers (>100MB) and infrequently
                if expected_size > 100 * 1024 * 1024 and len(received_data) >= next_progress_threshold:
                    logger.info(f"📥 Progress: {len(received_data)/1e6:.0f}/{expected_size/1e6:.0f} MB")
                    next_progress_threshold += 100 * 1024 * 1024  # Next 100MB
            
            # ✅ Always log completion with speed
            elapsed = time.time() - start_time
            speed_mbps = (expected_size / 1e6) / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Receive complete: {len(received_data)/1e6:.2f} MB in {elapsed:.2f}s ({speed_mbps:.0f} MB/s)")
            return bytes(received_data)
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Receive timeout after {timeout}s")
            return None
        except ConnectionResetError:
            logger.warning("⚠️  Peer connection reset (likely busy) - receive cancelled")
            return None
        except Exception as e:
            logger.error(f"❌ Receive failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    async def safe_connect(partner_id: str) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Create optimized connection with high-performance settings"""
        try:
            host, port_str = partner_id.split(':')
            port = int(port_str)
            
            # Use large buffer limits for high-throughput connections
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host, port,
                    limit=32*1024*1024  # 32MB buffer instead of default 64KB
                ), 
                timeout=10.0
            )
            
            # Apply socket optimizations immediately after connection
            sock = writer.get_extra_info('socket')
            if sock:
                # Critical TCP optimizations
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                
                if hasattr(socket, 'TCP_QUICKACK'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            
            return reader, writer
            
        except Exception as e:
            logging.error(f"Connection failed to {partner_id}: {e}")
            return None

import json
import time
import os
import threading
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from .network_utils import NetworkUtils

class GossipLogger:
    def __init__(self, node_id: str, global_rank: int, local_rank: int, data_parallel_rank: int, 
                 output_dir: Optional[str] = None):
        self.node_id = node_id
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.data_parallel_rank = data_parallel_rank
        
        # Get OUR node's actual identity
        self.local_ip, self.gossip_port = NetworkUtils.get_node_identity(local_rank)
        self.node_identity = f"{self.local_ip}:{self.gossip_port}"
        
        # Create gossip subdirectory in output directory
        if output_dir:
            log_dir = Path(output_dir) / "gossip"
        else:
            log_dir = Path("gossip")  # Fallback
        
        log_dir.mkdir(exist_ok=True)
        
        # Create per-node log file with simple name
        safe_identity = self.node_identity.replace(':', '_').replace('.', '_')
        self.log_file = log_dir / f"{safe_identity}_rank_{global_rank}.tsv"
        
        # Write TSV header
        if not self.log_file.exists():
            header = [
                "unix_timestamp", "node_identity", "global_rank", "dp_rank",
                "event_type", "step", "fitness", "correlation_id", 
                "peer_addr", "data_size_mb", "transfer_time_ms", "message"
            ]
            with open(self.log_file, 'w') as f:
                f.write('\t'.join(header) + '\n')
        
        self.lock = threading.Lock()
    
    def log_event(self, event_type: str, step: Optional[int] = None, 
                  fitness: Optional[float] = None, correlation_id: Optional[str] = None,
                  peer_addr: Optional[str] = None, data_size_bytes: Optional[int] = None,
                  transfer_time_ms: Optional[float] = None, message: str = ""):
        
        timestamp = time.time()
        data_size_mb = data_size_bytes / 1e6 if data_size_bytes else None
        
        values = [
            f"{timestamp:.6f}",
            self.node_identity,
            str(self.global_rank),
            str(self.data_parallel_rank),
            event_type,
            str(step) if step is not None else "",
            f"{fitness:.6f}" if fitness is not None else "",
            correlation_id or "",
            peer_addr or "",
            f"{data_size_mb:.3f}" if data_size_mb is not None else "",
            f"{transfer_time_ms:.2f}" if transfer_time_ms is not None else "",
            message
        ]
        
        with self.lock:
            with open(self.log_file, 'a') as f:
                f.write('\t'.join(values) + '\n')
    
    def generate_correlation_id(self) -> str:
        """Generate a short correlation ID for tracking exchanges"""
        return str(uuid.uuid4())[:8]

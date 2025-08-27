"""
Document-aware dataset classes used by both training and validation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import mmap


class SingleStreamDataset(Dataset):
    """
    Simple dataset that returns pre-created chunks.
    Used for single-stream data loading.
    """
    def __init__(self, data_path: str, chunk_size: int, total_chunks: int, 
                 rank: int, seed: int, shared_mmap=None):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.total_chunks = total_chunks
        self.rank = rank
        
        # Use shared mmap if provided, otherwise open the data file
        if shared_mmap is not None:
            self.mmap = shared_mmap
            self.file_size = len(self.mmap)
            self.data_file = None  # No file handle when using shared mmap
        else:
            # Open the data file
            self.data_file = open(data_path, 'rb')
            self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
            self.file_size = len(self.mmap)
        
        # Random start position based on rank and seed
        rng = np.random.RandomState(seed + rank)
        self.position = rng.randint(0, max(1, self.file_size - 1000))
        
        # Scan to document boundary
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1
        self.position = (self.position + 1) % self.file_size
        
        self.byte_buffer = []
        
    def __len__(self):
        return self.total_chunks
    
    def __getitem__(self, idx):
        """Get a chunk with document boundary handling"""
        chunk_data = []
        doc_ended = False
        actual_length = self.chunk_size
        
        for i in range(self.chunk_size):
            # Handle wrapping
            if self.position >= self.file_size:
                self.position = 0
            
            byte_val = self.mmap[self.position]
            self.position += 1
            
            if byte_val == 0x1e:  # Document boundary
                doc_ended = True
                # Record actual length and pad rest
                actual_length = i
                # Pad rest of chunk
                chunk_data.append(0)
                for j in range(i + 1, self.chunk_size):
                    chunk_data.append(0)
                break
            else:
                chunk_data.append(byte_val)
        
        # Fill if we didn't hit boundary
        while len(chunk_data) < self.chunk_size:
            if self.position >= self.file_size:
                self.position = 0
            byte_val = self.mmap[self.position]
            self.position += 1
            
            if byte_val == 0x1e:
                doc_ended = True
                chunk_data.append(0)
            else:
                chunk_data.append(byte_val)
        
        chunk = torch.tensor(chunk_data, dtype=torch.long)
        return chunk, doc_ended, actual_length
    
    def __del__(self):
        # Only close mmap and file if we own them (not shared)
        if hasattr(self, 'data_file') and self.data_file is not None:
            if hasattr(self, 'mmap'):
                self.mmap.close()
            self.data_file.close()


class DocumentStreamDataset(Dataset):
    """
    Document-aware streaming dataset for training.
    
    Key features:
    - Respects document boundaries (0x1e delimiter)
    - Each GPU starts at different random position
    - Resets model hidden state at document boundaries
    - Tracks per-GPU statistics (not global)
    """
    
    def __init__(self, data_path: str, chunk_size: int, rank: int, 
                 world_size: int, seed: int = 42):
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        
        # Open the data file
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)
        
        # Random starting position for this rank
        rng = np.random.RandomState(seed + rank)
        self.position = rng.randint(0, max(1, self.file_size - 1000))
        
        # Track statistics
        self.chunks_served = 0
        self.docs_completed = 0
        self.bytes_processed = 0
        self.wraps = 0
        
        # Scan forward to next document boundary to start clean
        self._scan_to_next_document()
        
        # Buffer for accumulating bytes
        self.byte_buffer = []
    
    def __len__(self):
        return 1_000_000_000  # Effectively infinite
    
    def _scan_to_next_document(self):
        """Scan forward to the start of the next document"""
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1
        
        # Skip the delimiter itself
        if self.position < self.file_size:
            self.position += 1
        else:
            # Wrapped around
            self.position = 0
            self.wraps += 1
    
    def get_next_chunk(self):
        """Get next chunk - wrapper for compatibility with DocumentStreamWrapper"""
        return self.__getitem__(0)
    
    def __getitem__(self, idx):
        """
        Returns: (chunk_tensor, is_final_chunk_in_doc, actual_chunk_length)
        
        IMPORTANT: Always returns fixed-size tensors for CUDA graph compatibility.
        Partial chunks are padded with zeros, and actual_length indicates valid data.
        """
        while len(self.byte_buffer) < self.chunk_size:
            # Check if we need to wrap
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1
            
            byte_val = self.mmap[self.position]
            self.position += 1
            self.bytes_processed += 1
            
            # Check for document boundary
            if byte_val == 0x1e:
                self.docs_completed += 1
                
                if len(self.byte_buffer) > 0:
                    # Partial chunk at document boundary - PAD to maintain fixed size
                    actual_length = len(self.byte_buffer)
                    
                    # Create full-sized chunk with padding
                    chunk = torch.zeros(self.chunk_size, dtype=torch.long)
                    chunk[:actual_length] = torch.tensor(self.byte_buffer, dtype=torch.long)
                    self.byte_buffer = []
                    
                    return chunk, True, actual_length
                else:
                    # Empty buffer at document start, skip delimiter
                    continue
            else:
                self.byte_buffer.append(byte_val)
        
        # Full chunk
        chunk = torch.tensor(self.byte_buffer[:self.chunk_size], dtype=torch.long)
        self.byte_buffer = self.byte_buffer[self.chunk_size:]
        
        return chunk, False, self.chunk_size
    
    def get_stats(self):
        """Return current dataset statistics"""
        return {
            'chunks_served': self.chunks_served,
            'documents_processed': self.docs_completed,  # Use consistent naming with train.py
            'docs_completed': self.docs_completed,  # Keep for backwards compatibility
            'bytes_processed': self.bytes_processed,
            'file_wraps': self.wraps,  # Match train.py naming
            'wraps': self.wraps,  # Keep for backwards compatibility
            'current_position': self.position,
            'position': self.position
        }
    
    def __del__(self):
        # Only close mmap and file if we own them (not shared)
        if hasattr(self, 'data_file') and self.data_file is not None:
            if hasattr(self, 'mmap'):
                self.mmap.close()
            self.data_file.close()
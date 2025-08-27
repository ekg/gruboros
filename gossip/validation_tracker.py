import numpy as np
import torch
from collections import deque
from typing import Optional, List, Tuple
import statistics
import time

class ValidationTracker:
    """Tracks model fitness using training-like validation"""
    
    def __init__(self, data_path: str, chunk_size: int, batch_size: int,
                 validation_interval: int = 10000,
                 validation_batches: int = 8,
                 window_size: int = 10):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.validation_interval = validation_interval
        self.validation_batches = validation_batches
        # Validation sequences are chunk_size long, same as training
        self.sequence_length = chunk_size
        
        self.mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.file_size = len(self.mmap)
        
        # Store validation results
        self.validation_losses = deque(maxlen=window_size)
        self.current_fitness = float('inf')
        self.last_validation_step = 0
        
    def should_validate(self, step: int) -> bool:
        """Check if it's time to run validation"""
        return step > 0 and step % self.validation_interval == 0
    
    def run_validation(self, model: torch.nn.Module, step: int, seed: int) -> float:
        """Run training-like validation and return mean loss"""
        losses, _ = self._evaluate_sequences(model, seed, return_document_losses=False)
        mean_loss = np.mean(losses)
        
        # Update our fitness tracking - use most recent value directly
        self.validation_losses.append(mean_loss)  # Keep history for analysis
        self.current_fitness = mean_loss  # Use current result, not median
        self.last_validation_step = step
        
        return self.current_fitness
    
    def evaluate_for_gossip(self, model: torch.nn.Module, seed: int) -> np.ndarray:
        """Same validation for gossip comparison - returns array of batch losses"""
        batch_losses, _ = self._evaluate_sequences(model, seed, return_document_losses=True)
        return batch_losses
    
    def _evaluate_sequences(self, model: torch.nn.Module, seed: int, return_document_losses: bool = False) -> Tuple:
        """Batched validation that matches training batch size for torch.compile compatibility
        
        Returns:
            - sequence_losses: array of per-sequence average losses (for fitness)
            - document_losses: array of per-document average losses (for t-test)
        """
        model.eval()
        device = next(model.parameters()).device
        rng = np.random.RandomState(seed)
        
        all_sequence_losses = []
        all_document_losses = []
        
        # Initialize batch_size parallel streams at random positions (like training)
        # Each stream will read sequentially through all validation batches
        positions = []
        for i in range(self.batch_size):
            start_pos = rng.randint(0, max(1, self.file_size - 1000))  # Leave some buffer
            # Scan to next document boundary
            while start_pos < self.file_size and self.mmap[start_pos] != 0x1e:
                start_pos += 1
            positions.append((start_pos + 1) % self.file_size)
        
        # Initialize hidden states - these persist across batches
        hidden_states = None
        conv_buffers = None
        
        # Process multiple batches contiguously
        num_batches = self.validation_batches
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                
                # Process one chunk for each stream
                batch_chunks = []
                actual_lengths = []
                is_doc_end = []
                
                for seq_idx in range(self.batch_size):
                    # Collect one chunk from this stream's current position
                    chunk_data = []
                    doc_ended = False
                    actual_len = self.chunk_size
                    
                    for i in range(self.chunk_size):
                        if positions[seq_idx] >= self.file_size:
                            positions[seq_idx] = 0  # Wrap around
                        
                        byte_val = int(self.mmap[positions[seq_idx]])
                        positions[seq_idx] += 1
                        
                        if byte_val == 0x1e:  # Document boundary
                            doc_ended = True
                            # Record actual length before padding
                            if actual_len == self.chunk_size:
                                actual_len = i + 1
                            # Continue to scan past boundary but pad with zeros
                            chunk_data.append(0)  # Pad rest of chunk
                        else:
                            chunk_data.append(byte_val)
                    
                    batch_chunks.append(torch.tensor(chunk_data, dtype=torch.long))
                    actual_lengths.append(actual_len)
                    is_doc_end.append(doc_ended)
                    
                
                # Stack into batch tensors
                chunks_tensor = torch.stack(batch_chunks).to(device)  # [batch_size, chunk_size]
                lengths_tensor = torch.tensor(actual_lengths, dtype=torch.long, device=device)
                doc_end_mask = torch.tensor(is_doc_end, dtype=torch.bool, device=device)
                
                # Forward pass with batched data
                result = model(
                    chunks_tensor,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=hidden_states,
                    prev_conv_buffers=conv_buffers,
                    actual_length=lengths_tensor
                )
                
                # Unpack result
                if isinstance(result, tuple) and len(result) == 2:
                    loss, (next_hidden_states, next_conv_buffers) = result
                else:
                    loss = result
                    next_hidden_states = None
                    next_conv_buffers = None
                
                # Record loss for this batch
                all_sequence_losses.append(loss.item())
                
                # Handle hidden state resets based on document boundaries
                if next_hidden_states:
                    # Create appropriate mask shape based on hidden state dimensions
                    if isinstance(next_hidden_states, list):
                        # List of hidden states from multiple layers
                        reset_mask = doc_end_mask.view(-1, 1)  # [B, 1] for 2D states
                        hidden_states = [h.detach() * (~reset_mask) for h in next_hidden_states]
                    else:
                        # Single hidden state
                        if next_hidden_states.dim() == 2:
                            reset_mask = doc_end_mask.view(-1, 1)  # [B, 1]
                        else:
                            reset_mask = doc_end_mask.view(-1, 1, 1)  # [B, 1, 1] for 3D
                        hidden_states = next_hidden_states.detach() * (~reset_mask)
                else:
                    hidden_states = None
                
                if next_conv_buffers:
                    conv_reset_mask = doc_end_mask.view(-1, 1, 1, 1)
                    conv_buffers = [b.detach() * (~conv_reset_mask) for b in next_conv_buffers]
                else:
                    conv_buffers = None
        
        model.train()
        
        # Return batch losses (no document tracking in streaming mode)
        return np.array(all_sequence_losses), None
    
    def get_fitness(self) -> float:
        """Return current validation loss (most recent)"""
        return self.current_fitness
    
    def inherit_fitness(self, source_fitness: float):
        """When receiving weights, inherit the source's validation fitness"""
        # Don't clear history, just update current fitness
        self.validation_losses.append(source_fitness)
        self.current_fitness = source_fitness
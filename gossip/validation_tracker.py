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
                 window_size: int = 10):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.validation_interval = validation_interval
        # Fixed validation length - process for ~8k tokens per sequence  
        self.sequence_length = 8192
        
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
        """Same validation for gossip comparison - returns array of document losses"""
        _, document_losses = self._evaluate_sequences(model, seed, return_document_losses=True)
        return np.array(document_losses)
    
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
        
        # Process just one batch for validation
        num_batches = 1
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Always process exactly batch_size sequences
                actual_batch_size = self.batch_size
                
                # Initialize batch of sequences at random positions
                positions = [rng.randint(0, max(1, self.file_size - self.sequence_length)) 
                            for _ in range(actual_batch_size)]
                
                # Scan each to next document boundary
                for i in range(actual_batch_size):
                    while positions[i] < self.file_size and self.mmap[positions[i]] != 0x1e:
                        positions[i] += 1
                    positions[i] = (positions[i] + 1) % self.file_size
                
                # No padding needed - we always process exactly batch_size sequences
                
                # Process batch
                hidden_states = None  # Will be list of [batch_size, ...] tensors
                conv_buffers = None
                bytes_processed = [0] * self.batch_size
                doc_losses_per_seq = [[] for _ in range(actual_batch_size)]
                current_doc_chunks = [[] for _ in range(actual_batch_size)]
                
                while any(b < self.sequence_length for b in bytes_processed[:actual_batch_size]):
                    # Collect chunks for entire batch
                    batch_chunks = []
                    actual_lengths = []
                    is_doc_end = []
                    
                    for seq_idx in range(self.batch_size):
                        if seq_idx >= actual_batch_size or bytes_processed[seq_idx] >= self.sequence_length:
                            # Padding sequence or completed sequence
                            batch_chunks.append(torch.zeros(self.chunk_size, dtype=torch.long))
                            actual_lengths.append(self.chunk_size)
                            is_doc_end.append(False)
                        else:
                            # Collect chunk for this sequence
                            chunk_data = []
                            doc_ended = False
                            
                            while len(chunk_data) < self.chunk_size and bytes_processed[seq_idx] < self.sequence_length:
                                if positions[seq_idx] >= self.file_size:
                                    positions[seq_idx] = 0
                                
                                byte_val = int(self.mmap[positions[seq_idx]])
                                positions[seq_idx] += 1
                                bytes_processed[seq_idx] += 1
                                
                                if byte_val == 0x1e:  # Document boundary
                                    doc_ended = True
                                    if current_doc_chunks[seq_idx]:
                                        # Calculate document loss
                                        doc_loss = sum(l * t for l, t in current_doc_chunks[seq_idx]) / \
                                                  sum(t for _, t in current_doc_chunks[seq_idx])
                                        doc_losses_per_seq[seq_idx].append(doc_loss)
                                        current_doc_chunks[seq_idx] = []
                                    break
                                else:
                                    chunk_data.append(byte_val)
                            
                            # Pad chunk to full size
                            actual_len = len(chunk_data)
                            if actual_len < self.chunk_size:
                                chunk_data.extend([0] * (self.chunk_size - actual_len))
                            
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
                    
                    # Record losses for actual sequences
                    loss_val = loss.item()
                    for seq_idx in range(actual_batch_size):
                        if bytes_processed[seq_idx] <= self.sequence_length and actual_lengths[seq_idx] > 0:
                            current_doc_chunks[seq_idx].append((loss_val, actual_lengths[seq_idx]))
                    
                    # Handle hidden state resets based on document boundaries
                    if next_hidden_states:
                        reset_mask = doc_end_mask.view(-1, 1, 1)
                        hidden_states = [h.detach() * (~reset_mask) for h in next_hidden_states]
                    
                    if next_conv_buffers:
                        conv_reset_mask = doc_end_mask.view(-1, 1, 1, 1)
                        conv_buffers = [b.detach() * (~conv_reset_mask) for b in next_conv_buffers]
                
                # Finalize any incomplete documents
                for seq_idx in range(actual_batch_size):
                    if current_doc_chunks[seq_idx]:
                        doc_loss = sum(l * t for l, t in current_doc_chunks[seq_idx]) / \
                                  sum(t for _, t in current_doc_chunks[seq_idx])
                        doc_losses_per_seq[seq_idx].append(doc_loss)
                
                # Calculate sequence-level losses for this batch
                batch_sequence_losses = []
                for seq_idx in range(actual_batch_size):
                    if doc_losses_per_seq[seq_idx]:
                        batch_sequence_losses.append(np.mean(doc_losses_per_seq[seq_idx]))
                
                all_sequence_losses.extend(batch_sequence_losses)
                
                # Collect all document losses
                for seq_losses in doc_losses_per_seq:
                    all_document_losses.extend(seq_losses)
        
        model.train()
        
        if return_document_losses:
            return np.array(all_sequence_losses), all_document_losses
        else:
            return np.array(all_sequence_losses), None
    
    def get_fitness(self) -> float:
        """Return current validation loss (most recent)"""
        return self.current_fitness
    
    def inherit_fitness(self, source_fitness: float):
        """When receiving weights, inherit the source's validation fitness"""
        # Don't clear history, just update current fitness
        self.validation_losses.append(source_fitness)
        self.current_fitness = source_fitness
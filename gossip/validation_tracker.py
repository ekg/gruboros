import numpy as np
import torch
from collections import deque
from typing import Optional, List
import statistics
import time

class ValidationTracker:
    """Tracks model fitness using training-like validation"""
    
    def __init__(self, data_path: str, chunk_size: int, 
                 validation_interval: int = 10000,
                 num_sequences: int = 32,
                 sequence_length: int = 8192,
                 window_size: int = 10):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.validation_interval = validation_interval
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        
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
        losses = self._evaluate_sequences(model, seed)
        mean_loss = np.mean(losses)
        
        # Update our fitness tracking - use most recent value directly
        self.validation_losses.append(mean_loss)  # Keep history for analysis
        self.current_fitness = mean_loss  # Use current result, not median
        self.last_validation_step = step
        
        return self.current_fitness
    
    def evaluate_for_gossip(self, model: torch.nn.Module, seed: int) -> np.ndarray:
        """Same validation for gossip comparison - returns array of losses"""
        return self._evaluate_sequences(model, seed)
    
    def _evaluate_sequences(self, model: torch.nn.Module, seed: int) -> np.ndarray:
        """Core validation logic that mimics training exactly"""
        model.eval()
        device = next(model.parameters()).device
        rng = np.random.RandomState(seed)
        losses = []
        
        with torch.no_grad():
            for seq_idx in range(self.num_sequences):
                # Start at random position
                position = rng.randint(0, max(1, self.file_size - self.sequence_length))
                
                # Scan to next document boundary
                while position < self.file_size and self.mmap[position] != 0x1e:
                    position += 1
                position = (position + 1) % self.file_size  # Skip delimiter
                
                # Process one sequence
                hidden_state = None
                sequence_losses = []
                bytes_processed = 0
                
                while bytes_processed < self.sequence_length:
                    # Collect chunk respecting document boundaries
                    chunk_data = []
                    
                    while len(chunk_data) < self.chunk_size and bytes_processed < self.sequence_length:
                        if position >= self.file_size:
                            position = 0
                        
                        byte_val = int(self.mmap[position])
                        position += 1
                        bytes_processed += 1
                        
                        if byte_val == 0x1e:  # Document boundary
                            if chunk_data:
                                # Process partial chunk before boundary
                                chunk = torch.tensor(chunk_data, dtype=torch.long).unsqueeze(0).to(device)
                                loss = model(chunk, return_loss=True, prev_hiddens=hidden_state)
                                sequence_losses.append((loss.item(), len(chunk_data)))
                                hidden_state = None  # Reset for new document
                                chunk_data = []
                        else:
                            chunk_data.append(byte_val)
                    
                    # Process full or final chunk
                    if chunk_data:
                        chunk = torch.tensor(chunk_data, dtype=torch.long).unsqueeze(0).to(device)
                        loss, next_hidden = model(chunk, return_loss=True, 
                                                prev_hiddens=hidden_state, 
                                                return_prev_hiddens=True)
                        sequence_losses.append((loss.item(), len(chunk_data)))
                        hidden_state = next_hidden
                
                # Compute weighted average loss for this sequence
                if sequence_losses:
                    total_loss = sum(loss * tokens for loss, tokens in sequence_losses)
                    total_tokens = sum(tokens for _, tokens in sequence_losses)
                    losses.append(total_loss / total_tokens)
        
        model.train()
        return np.array(losses)
    
    def get_fitness(self) -> float:
        """Return current validation loss (most recent)"""
        return self.current_fitness
    
    def inherit_fitness(self, source_fitness: float):
        """When receiving weights, inherit the source's validation fitness"""
        # Don't clear history, just update current fitness
        self.validation_losses.append(source_fitness)
        self.current_fitness = source_fitness
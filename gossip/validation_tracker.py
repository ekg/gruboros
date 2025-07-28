import numpy as np
import torch
from collections import deque
from typing import Optional, List
import statistics
import time

class ValidationTracker:
    """Tracks model fitness using periodic validation on random chunks"""
    
    def __init__(self, data_path: str, chunk_size: int, 
                 validation_interval: int = 100,
                 chunks_per_validation: int = 4,
                 window_size: int = 1000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.validation_interval = validation_interval
        self.chunks_per_validation = chunks_per_validation
        self.window_size = window_size
        
        self.mmap = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.file_size = len(self.mmap)
        self.max_start = self.file_size - self.chunk_size
        
        # Store individual validation losses
        self.validation_losses = deque(maxlen=window_size)
        self.current_fitness = float('inf')
        self.last_validation_step = 0
        self.total_validations = 0
        
    def should_validate(self, step: int) -> bool:
        """Check if it's time to run validation"""
        return step > 0 and step % self.validation_interval == 0
    
    def run_validation(self, model: torch.nn.Module, step: int, seed: int) -> float:
        """Run validation and return median of recent validations"""
        model.eval()
        
        # Use step + seed for reproducible randomness
        rng = np.random.RandomState(seed + step)
        
        # Sample random positions
        positions = rng.randint(0, self.max_start, size=self.chunks_per_validation)
        
        # Evaluate each chunk
        chunk_losses = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for pos in positions:
                chunk_data = self.mmap[pos:pos + self.chunk_size]
                chunk = torch.tensor(chunk_data, dtype=torch.long)
                chunk = chunk.unsqueeze(0).to(device, non_blocking=True)
                
                loss = model(chunk, return_loss=True)
                chunk_losses.append(loss.item())
        
        # Add all chunk losses to our window
        self.validation_losses.extend(chunk_losses)
        self.total_validations += 1
        
        # Update fitness if we have enough samples
        if len(self.validation_losses) >= 10:
            self.current_fitness = statistics.median(self.validation_losses)
        else:
            self.current_fitness = sum(self.validation_losses) / len(self.validation_losses)
        
        self.last_validation_step = step
        
        # Back to training mode
        model.train()
        
        return self.current_fitness
    
    def get_fitness(self) -> float:
        """Return current median validation loss"""
        return self.current_fitness
    
    def inherit_fitness(self, source_fitness: float):
        """When receiving weights, inherit the source's validation fitness"""
        self.validation_losses.clear()
        # Seed with a few copies to avoid instability
        self.validation_losses.extend([source_fitness] * self.chunks_per_validation)
        self.current_fitness = source_fitness
        
    def evaluate_positions(self, model: torch.nn.Module, positions: List[int]) -> np.ndarray:
        """Evaluate model on specific positions for gossip comparison"""
        model.eval()
        losses = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for pos in positions:
                chunk_data = self.mmap[pos:pos + self.chunk_size]
                chunk = torch.tensor(chunk_data, dtype=torch.long)
                chunk = chunk.unsqueeze(0).to(device, non_blocking=True)
                
                loss = model(chunk, return_loss=True)
                losses.append(loss.item())
        
        model.train()
        return np.array(losses)
import numpy as np
import statistics
from typing import Optional
from collections import deque

class FitnessTracker:
    def __init__(self, window_size: int = 1000):
        self.fitness_window = deque(maxlen=window_size)
        self.current_fitness = float('inf')
        self.step_count = 0
        
    def update(self, loss_value: float):
        """Update fitness using windowed median"""
        self.fitness_window.append(loss_value)
        
        # Calculate median if we have enough samples
        if len(self.fitness_window) >= 10:  # Minimum samples for stable median
            self.current_fitness = statistics.median(self.fitness_window)
        else:
            # Use mean during initial warmup
            self.current_fitness = sum(self.fitness_window) / len(self.fitness_window)
        
        self.step_count += 1
        
        # Log fitness updates occasionally for debugging
        if self.step_count % 10 == 0:  # Every 10 updates
            current_fitness = self.get_fitness()
            import logging
            logger = logging.getLogger('fitness_tracker')
            logger.debug(f"Fitness update: loss={loss_value:.4f}, fitness={current_fitness:.4f}, window_size={len(self.fitness_window)}")
    
    def get_fitness(self) -> float:
        """Return median loss (lower is better)"""
        return self.current_fitness
    
    def get_recent_loss(self) -> float:
        """Get current median loss"""
        return self.current_fitness
    
    def inherit_fitness(self, source_median_loss: float):
        """Inherit median loss from source model when completely overwritten"""
        # Clear current window and seed with source fitness
        self.fitness_window.clear()
        self.fitness_window.append(source_median_loss)
        self.current_fitness = source_median_loss
        
        import logging
        logger = logging.getLogger('fitness_tracker')
        logger.debug(f"Inherited median_loss {source_median_loss:.4f}")

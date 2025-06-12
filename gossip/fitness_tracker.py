import numpy as np
from typing import List

class FitnessTracker:
    def __init__(self, decay_factor: float = 0.95, max_history: int = 50):
        self.loss_history: List[float] = []
        self.decay_factor = decay_factor
        self.max_history = max_history
        
    def update(self, loss_value: float):
        """Update fitness based on recent loss"""
        self.loss_history.append(loss_value)
        if len(self.loss_history) > self.max_history:
            self.loss_history.pop(0)
    
    def get_fitness(self) -> float:
        """Calculate fitness as decaying weighted average of recent losses"""
        if not self.loss_history:
            return 1.0
            
        weights = [self.decay_factor ** i for i in range(len(self.loss_history))]
        weighted_losses = [w * loss for w, loss in zip(weights, self.loss_history)]
        avg_loss = sum(weighted_losses) / sum(weights)
        return 1.0 / (avg_loss + 1e-6)
    
    def get_recent_loss(self) -> float:
        """Get most recent loss"""
        return self.loss_history[-1] if self.loss_history else float('inf')

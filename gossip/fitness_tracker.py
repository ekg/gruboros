import numpy as np
from typing import Optional

class FitnessTracker:
    def __init__(self, decay_factor: float = 0.95):
        self.ema_loss: Optional[float] = None  # Rolling exponential moving average
        self.alpha = 1.0 - decay_factor  # Convert to EMA alpha
        self.step_count = 0
        
    def update(self, loss_value: float):
        """Update fitness based on recent loss using exponential moving average"""
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = self.alpha * loss_value + (1 - self.alpha) * self.ema_loss
        
        self.step_count += 1
        
        # Log fitness updates occasionally for debugging
        if self.step_count % 10 == 0:  # Every 10 updates
            current_fitness = self.get_fitness()
            import logging
            logger = logging.getLogger('fitness_tracker')
            logger.debug(f"Fitness update: loss={loss_value:.4f}, fitness={current_fitness:.4f}, ema_loss={self.ema_loss:.4f}")
    
    def get_fitness(self) -> float:
        """Calculate fitness as inverse of EMA loss"""
        if self.ema_loss is None:
            return 1.0
        return 1.0 / (self.ema_loss + 1e-6)
    
    def get_recent_loss(self) -> float:
        """Get current EMA loss"""
        return self.ema_loss if self.ema_loss is not None else float('inf')
    
    def inherit_fitness(self, source_fitness: float, source_ema_loss: Optional[float] = None):
        """Inherit fitness from source model when completely overwritten"""
        if source_ema_loss is not None:
            # Directly inherit the EMA loss (preserves learning history)
            self.ema_loss = source_ema_loss
        else:
            # Fallback: convert fitness back to loss
            self.ema_loss = 1.0 / (source_fitness + 1e-6) - 1e-6
        
        import logging
        logger = logging.getLogger('fitness_tracker')
        logger.debug(f"Inherited fitness {source_fitness:.4f} (ema_loss={self.ema_loss:.4f})")

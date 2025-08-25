"""Neural Arithmetic Unit GRU - Combines NAU with log-barrier dynamics for stable arithmetic learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NeuralArithmeticGate(nn.Module):
    """Learns to select between additive and multiplicative operations"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Gate networks
        self.W_hat = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.M_hat = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.G = nn.Linear(dim * 2, dim)  # Gate based on input and hidden
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Compute gates
        g = torch.sigmoid(self.G(torch.cat([x, h], dim=-1)))
        
        # Additive path (normal space)
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.W_hat)  # Weights in [-1, 1]
        a = torch.matmul(x, W.T)
        
        # Multiplicative path (log space)  
        M = torch.tanh(self.M_hat)  # Weights for exponents
        m = torch.matmul(torch.log(torch.abs(x) + 1e-8), M.T)
        
        # Gate between them
        return g * a + (1 - g) * torch.exp(m)


class LogBarrierDynamics(nn.Module):
    """Maintains numerical stability through repulsive barriers"""
    def __init__(self, dim: int, min_log: float = -10, max_log: float = 10):
        super().__init__()
        self.min_log = min_log
        self.max_log = max_log
        self.barrier_strength = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, log_h: torch.Tensor) -> torch.Tensor:
        # Repulsive force from boundaries
        lower_barrier = -self.barrier_strength / (log_h - self.min_log + 1e-3)
        upper_barrier = self.barrier_strength / (self.max_log - log_h + 1e-3)
        
        # Apply barriers (stronger as we approach limits)
        log_h = log_h + lower_barrier + upper_barrier
        
        # Hard clamp as safety
        return torch.clamp(log_h, self.min_log + 0.1, self.max_log - 0.1)


class NAU_GRU(nn.Module):
    """minGRU variant with Neural Arithmetic Units and log-barrier dynamics"""
    def __init__(self, dim: int, expansion_factor: float = 1.5, 
                 use_nau: bool = True, use_barriers: bool = True,
                 barrier_min: float = -10, barrier_max: float = 10):
        super().__init__()
        self.dim = dim
        self.use_nau = use_nau
        self.use_barriers = use_barriers
        self.expansion_factor = expansion_factor
        
        dim_inner = int(dim * expansion_factor)
        self.dim_inner = dim_inner
        
        # Standard minGRU components
        self.to_hidden_and_gate = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        
        # NAU components
        if use_nau:
            self.nau = NeuralArithmeticGate(dim_inner)
            
        # Log barrier dynamics
        if use_barriers:
            self.barrier = LogBarrierDynamics(dim_inner, barrier_min, barrier_max)
            
        # Initialize
        nn.init.xavier_uniform_(self.to_hidden_and_gate.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward_sequential(self, x: torch.Tensor, 
                         prev_log_hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential forward for training with NAU"""
        batch_size, seq_len, _ = x.shape
        
        if prev_log_hidden is None:
            prev_log_hidden = torch.zeros(batch_size, self.dim_inner, device=x.device)
        
        outputs = []
        log_hidden = prev_log_hidden
        
        for t in range(seq_len):
            x_t = x[:, t]
            hidden_t, gate_t = self.to_hidden_and_gate(x_t).chunk(2, dim=-1)
            
            if self.use_nau:
                # NAU path: learns arithmetic operations
                h_normal = torch.exp(log_hidden)
                nau_update = self.nau(hidden_t, h_normal)
                log_new = torch.log(nau_update.abs() + 1e-8)
            else:
                # Standard minGRU path
                log_new = self.log_g(hidden_t)
            
            # Mixing gate
            gate_sigmoid = torch.sigmoid(gate_t)
            
            # Update in log space
            log_hidden = (1 - gate_sigmoid) * log_hidden + gate_sigmoid * log_new
            
            # Apply barriers to prevent explosion/collapse
            if self.use_barriers:
                log_hidden = self.barrier(log_hidden)
            
            # Output projection
            out = self.to_out(torch.exp(log_hidden))
            outputs.append(out)
        
        return torch.stack(outputs, dim=1), log_hidden
    
    def forward_scan(self, x: torch.Tensor, 
                    prev_hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel scan for inference - fallback to original minGRU"""
        # For now, fall back to sequential during scan
        # TODO: Implement proper parallel scan for NAU
        if prev_hidden is not None:
            prev_log_hidden = torch.log(prev_hidden.abs() + 1e-8)
        else:
            prev_log_hidden = None
        return self.forward_sequential(x, prev_log_hidden)
        
    def forward(self, x: torch.Tensor, 
                prev_hidden: Optional[torch.Tensor] = None,
                return_prev_hiddens: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main forward pass"""
        if self.training and self.use_nau:
            # Use sequential for NAU during training
            if prev_hidden is not None:
                prev_log_hidden = torch.log(prev_hidden.abs() + 1e-8)
            else:
                prev_log_hidden = None
            out, log_hidden = self.forward_sequential(x, prev_log_hidden)
            # Convert back to normal space for compatibility
            return out, torch.exp(log_hidden)
        else:
            # Use scan for inference (currently falls back to sequential)
            return self.forward_scan(x, prev_hidden)
    
    @staticmethod
    def log_g(x: torch.Tensor) -> torch.Tensor:
        """Log-space activation from original minGRU"""
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
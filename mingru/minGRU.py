# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    THE KEY CHANGE: This now returns log-space values, NOT exponentiated!
    This is mathematically correct - the scan operates in log space and
    should return log space values.
    """
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h  # <-- No .exp() here! Stay in log space!

# appendix B.3

def g(x):
    """Only used for reference - we use log_g in practice"""
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    """Log-space version of g(x) activation"""
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1

class minGRU(Module):
    """
    MinGRU with pure log-space hidden state handling.
    Hidden states are ALWAYS in log space when passed between time steps.
    """
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            prev_hidden: Previous hidden state in LOG SPACE (or None)
            return_next_prev_hidden: Whether to return the next hidden state
            
        Returns:
            out: Output in normal space (for next layer)
            next_prev_hidden: Next hidden state in LOG SPACE
        """
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # Sequential case - need log-space interpolation
            log_hidden = log_g(hidden)
            gate_sigmoid = gate.sigmoid()
            
            if exists(prev_hidden):
                # prev_hidden is ALREADY IN LOG SPACE - no conversion needed!
                # We need to compute: log(prev * (1-gate) + hidden * gate)
                
                # Convert gate to log probabilities
                eps = 1e-20  # for numerical stability
                log_gate = torch.log(gate_sigmoid + eps)
                log_one_minus_gate = torch.log(1 - gate_sigmoid + eps)
                
                # Use log-sum-exp for the interpolation
                # log(exp(log_prev * (1-g)) + exp(log_hidden * g))
                log_out = torch.logaddexp(
                    log_one_minus_gate + prev_hidden,
                    log_gate + log_hidden
                )
            else:
                # No previous state: out = hidden * gate
                # In log space: log(out) = log(hidden) + log(gate)
                log_out = log_hidden + torch.log(gate_sigmoid + 1e-20)
            
            # Hidden state STAYS in log space
            next_prev_hidden = log_out
            
            # Only exponentiate for the output projection
            out = log_out.exp()
            
        else:
            # Parallel case - this was already mostly correct!
            log_coeffs = -F.softplus(gate)  # log(1 - sigmoid(gate))
            
            log_z = -F.softplus(-gate)  # log(sigmoid(gate))
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                # Beautiful: no .log() needed because prev_hidden is already in log space!
                # This was the main source of numerical instability - now fixed
                log_values = torch.cat((prev_hidden, log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            # Returns log-space values (our key change to the scan function)
            log_out = heinsen_associative_scan_log(log_coeffs, log_values)
            log_out = log_out[:, -seq_len:]
            
            # Hidden state STAYS in log space
            next_prev_hidden = log_out[:, -1:]
            
            # Only exponentiate for the output projection
            out = log_out.exp()

        # Apply output projection to normal-space values
        # (Linear layers require normal space - this is unavoidable)
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from mingru.minGRU import minGRU
from mingru.nau_gru_cell import NAU_GRU

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class RMSNorm(Module):
    """BFloat16-aware RMS normalization that avoids upcasting"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize in float32 for numerical stability, but immediately cast back
        normed = F.normalize(x, dim=-1, eps=1e-5)
        return normed * self.scale * self.gamma

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalConv1d(Module):
    """Causal convolution with learnable scaling factor (ReZero/LayerScale style)"""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding_size = kernel_size - 1
        
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                             bias=False, padding=0)
        # PyTorch will initialize this automatically
        
        # Learnable scaling parameter, initialized small for gradual learning
        self.scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x, prev_buffer=None):
        batch_size, seq_len, dim = x.shape
        x_conv = x.transpose(1, 2).contiguous()
        
        if prev_buffer is not None and prev_buffer.numel() > 0:
            x_padded = torch.cat([prev_buffer, x_conv], dim=2)
        else:
            x_padded = F.pad(x_conv, (self.padding_size, 0), value=0.)
        
        out = self.conv(x_padded)
        
        if seq_len >= self.padding_size:
            next_buffer = x_conv[:, :, -self.padding_size:].detach().contiguous()
        else:
            next_buffer = x_padded[:, :, -self.padding_size:].detach().contiguous()
        
        # Apply learnable scaling before returning
        out = out.transpose(1, 2).contiguous()
        return out * self.scale, next_buffer

# main class

class minLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        expansion = 1.5,
        conv_kernel_size = None,  # None = no conv, or specify size (4, 8, 16, etc.)
        use_lstm = None,  # Kept for backward compatibility but ignored
        enable_conv = None,  # Deprecated - for backwards compatibility only
        dropout = 0.,
        use_nau = False,  # Enable Neural Arithmetic Units
        use_barriers = True,  # Enable log-barrier dynamics
        barrier_min = -10,  # Minimum log value before barrier
        barrier_max = 10  # Maximum log value before barrier
    ):
        super().__init__()
        
        # Handle backwards compatibility
        if enable_conv is not None:
            # Old style parameter - convert to new style
            if enable_conv:
                conv_kernel_size = conv_kernel_size if conv_kernel_size != 3 else 3
            else:
                conv_kernel_size = None
        
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        # Choose RNN class based on use_nau
        if use_nau:
            min_rnn_klass = NAU_GRU
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_nau': use_nau,
                'use_barriers': use_barriers,
                'barrier_min': barrier_min,
                'barrier_max': barrier_max
            }
        else:
            min_rnn_klass = minGRU
            rnn_kwargs = {'expansion_factor': expansion}

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalConv1d(dim, conv_kernel_size) if conv_kernel_size else None,
                RMSNorm(dim),
                min_rnn_klass(dim, **rnn_kwargs),
                RMSNorm(dim) if ff_mult > 0 else None,
                FeedForward(dim, mult = ff_mult) if ff_mult > 0 else None,
                nn.Dropout(dropout) if dropout > 0. else None
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.can_cache = (conv_kernel_size is None)
        
        # Store dimensions for initialization
        self.dim = dim
        self.depth = depth
        
        # Initialize weights with properly scaled standard deviations
        self._initialize_weights()

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_hiddens = False,
        prev_hiddens = None,
        prev_conv_buffers = None,  # New parameter for conv buffers
        actual_length = None  # For masking padded chunks at doc boundaries
    ):
        """
        Forward pass with support for both RNN hidden states and conv buffers.
        
        Conv buffers maintain the last (kernel_size - 1) tokens from the previous chunk,
        allowing convolutional layers to see across chunk boundaries correctly.
        This is separate from RNN hidden states which maintain sequential memory.
        
        Both are reset at document boundaries to maintain document independence.
        """

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle previous hiddens for single-step autoregressive decoding
        # This logic is ONLY for inference, not for training with loss calculation.
        if exists(prev_hiddens) and not return_loss:
            x = x[:, -1:]

        next_prev_hiddens = []
        next_conv_buffers = []
        
        # Handle tuple format from previous generation step
        if isinstance(prev_hiddens, tuple) and len(prev_hiddens) == 2:
            prev_hiddens, prev_conv_buffers = prev_hiddens
        
        prev_hiddens = iter(default(prev_hiddens, []))
        prev_conv_buffers = iter(default(prev_conv_buffers, []))

        for conv, norm, mingru, ff_norm, ff, dropout in self.layers:

            # conv

            if exists(conv):
                # Get previous buffer for this layer's conv
                prev_buffer = next(prev_conv_buffers, None)
                # Apply conv with buffer for continuity across chunks
                conv_out, next_buffer = conv(x, prev_buffer)
                x = conv_out + x
                # Store buffer in transposed format [B, D, L] for efficiency
                next_conv_buffers.append(next_buffer)

            # min gru

            prev_hidden = next(prev_hiddens, None)

            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden = True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward

            if exists(ff) and exists(ff_norm):
                x = ff(ff_norm(x)) + x
            
            # dropout
            
            if exists(dropout):
                x = dropout(x)

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            # Return both RNN hiddens and conv buffers for inference
            return logits, (next_prev_hiddens, next_conv_buffers)

        # Vectorized loss masking for batched padded sequences
        labels_masked = labels.clone()
        if actual_length is not None and torch.is_tensor(actual_length):
            seq_len = labels.size(1)
            # Create arange on the SAME device as labels to prevent device mismatch
            arange = torch.arange(seq_len, device=labels.device)[None, :]
            # Create a boolean mask for tokens to be ignored
            mask = arange >= (actual_length - 1)[:, None]
            labels_masked[mask] = -100
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels_masked,
            ignore_index=-100
        )

        # Modified return logic for TBPTT
        if not return_prev_hiddens:
            return loss
        
        # Return both RNN hiddens and conv buffers for training
        return loss, (next_prev_hiddens, next_conv_buffers)
        
    def _initialize_weights(self):
        """
        Initialize weights with zero initialization for final layers in residual blocks.
        This ensures each block starts as an identity function, preventing training instability.
        """
        # Calculate base standard deviation based on model dimension
        std = 0.02 / math.sqrt(self.dim)
        
        # Orthogonal initialization for embeddings - maximal separation between tokens
        nn.init.orthogonal_(self.token_emb.weight)
        # Scale down for gradient stability
        with torch.no_grad():
            self.token_emb.weight.mul_(0.1)
        
        # Initialize output projection carefully
        nn.init.normal_(self.to_logits.weight, mean=0.0, std=std)
        
        # Initialize internal layers
        for layer in self.layers:
            # Initialize minGRU/minLSTM weights
            min_rnn = layer[2]
            
            # Handle minGRU initialization
            if hasattr(min_rnn, 'to_hidden_and_gate'):
                # Input-facing layer gets normal initialization
                nn.init.normal_(min_rnn.to_hidden_and_gate.weight, mean=0.0, std=std)
                # Output-facing layer gets zero initialization for identity function
                if hasattr(min_rnn, 'to_out') and isinstance(min_rnn.to_out, nn.Linear):
                    nn.init.constant_(min_rnn.to_out.weight, 0.)
                    if min_rnn.to_out.bias is not None:
                        nn.init.constant_(min_rnn.to_out.bias, 0.)
            
            # Handle minLSTM initialization
            if hasattr(min_rnn, 'to_hidden_and_f_i_gate'):
                nn.init.normal_(min_rnn.to_hidden_and_f_i_gate.weight, mean=0.0, std=std)
                if hasattr(min_rnn, 'to_output_gate'):
                    nn.init.normal_(min_rnn.to_output_gate.weight, mean=0.0, std=std)
                if hasattr(min_rnn, 'to_out') and isinstance(min_rnn.to_out, nn.Linear):
                    nn.init.constant_(min_rnn.to_out.weight, 0.)
                    if min_rnn.to_out.bias is not None:
                        nn.init.constant_(min_rnn.to_out.bias, 0.)
            
            # Initialize feedforward layers
            ff = layer[4]
            if exists(ff) and isinstance(ff, nn.Sequential):
                # First FF layer gets normal initialization
                if len(ff) > 0 and isinstance(ff[0], nn.Linear):
                    nn.init.normal_(ff[0].weight, mean=0.0, std=std)
                # Final FF layer gets zero initialization for identity function
                if len(ff) > 2 and isinstance(ff[2], nn.Linear):
                    nn.init.constant_(ff[2].weight, 0.)
                    if ff[2].bias is not None:
                        nn.init.constant_(ff[2].bias, 0.)
            
            # Initialize conv layer - use PyTorch defaults as in the paper
            conv = layer[0]  # CausalConv1d if it exists
            if conv is not None:
                pass  # Use PyTorch's default initialization

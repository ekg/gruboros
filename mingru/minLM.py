import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm

from mingru.minGRU import minGRU

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    """Optimized causal convolution with efficient buffering."""
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding_size = kernel_size - 1
        
        # Convolution layers
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                   groups=dim, bias=False, padding=0)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x, prev_buffer=None):
        """
        Optimized forward with minimal memory operations.
        
        Key optimizations:
        1. Single transpose at start and end
        2. Contiguous memory for buffer
        3. No detach() in critical path
        """
        batch_size, seq_len, dim = x.shape
        
        # Single transpose
        x_conv = x.transpose(1, 2).contiguous()  # [B, D, L]
        
        # Efficient padding/buffering
        if prev_buffer is not None and prev_buffer.numel() > 0:
            # prev_buffer should already be [B, D, padding_size] and contiguous
            x_padded = torch.cat([prev_buffer, x_conv], dim=2)
        else:
            # First chunk - use zero padding
            x_padded = F.pad(x_conv, (self.padding_size, 0), value=0.)
        
        # Apply convolutions in sequence (fused operations)
        out = self.depthwise(x_padded)
        out = self.pointwise(out)
        
        # Prepare next buffer - make contiguous for next iteration
        if seq_len >= self.padding_size:
            # Take last padding_size elements from input
            next_buffer = x_conv[:, :, -self.padding_size:].clone()
        else:
            # Short sequence - need to handle carefully
            next_buffer = x_padded[:, :, -self.padding_size:].clone()
        
        # Single transpose back
        out = out.transpose(1, 2).contiguous()  # [B, L, D]
        
        return out, next_buffer

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
        dropout = 0.
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

        min_rnn_klass = minGRU

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if conv_kernel_size else None,
                RMSNorm(dim),
                min_rnn_klass(dim, expansion_factor = expansion),
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
        prev_conv_buffers = None  # New parameter for conv buffers
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

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
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
        
        # Initialize embedding with smaller std
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)
        
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

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

class CausalConv1d(Module):
    """
    Two-stage causal convolution block matching minGRU/FF pattern:
    1. Convolution stage (non-zero weights for gradient flow)
    2. Projection stage (zero weights for exact identity)
    """
    def __init__(self, dim, kernel_size=16):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding_size = kernel_size - 1
        
        # Stage 1: Convolution with full channel interaction
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                             bias=False, padding=0)
        
        # Stage 2: Projection layer (like minGRU to_out and FF final layer)
        self.proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, prev_buffer=None):
        """
        Forward pass with two-stage processing for exact identity initialization.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            prev_buffer: Buffer from previous chunk [batch, dim, padding_size] or None
        
        Returns:
            output: Processed output [batch, seq_len, dim]
            next_buffer: Buffer for next chunk [batch, dim, padding_size]
        """
        batch_size, seq_len, dim = x.shape
        
        # Transpose for conv1d: [batch, seq_len, dim] -> [batch, dim, seq_len]
        x_conv = x.transpose(1, 2).contiguous()
        
        # Apply causal padding using buffer or zeros
        if prev_buffer is not None and prev_buffer.numel() > 0:
            # Concatenate buffer (past context) with current input
            x_padded = torch.cat([prev_buffer, x_conv], dim=2)
        else:
            # First chunk or after reset - pad with zeros
            x_padded = F.pad(x_conv, (self.padding_size, 0), value=0.)
        
        # Stage 1: Convolution (non-zero weights enable gradient flow)
        out = self.conv(x_padded)  # [batch, dim, seq_len]
        
        # Transpose back for projection: [batch, dim, seq_len] -> [batch, seq_len, dim]
        out = out.transpose(1, 2).contiguous()
        
        # Stage 2: Projection (zero weights create exact identity)
        out = self.proj(out)
        
        # Extract buffer for next chunk (last padding_size timesteps)
        if seq_len >= self.padding_size:
            next_buffer = x_conv[:, :, -self.padding_size:].detach().contiguous()
        else:
            # Handle short sequences by taking from padded input
            next_buffer = x_padded[:, :, -self.padding_size:].detach().contiguous()
        
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
                CausalConv1d(dim, conv_kernel_size) if conv_kernel_size else None,
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

        # Handle masking for padded sequences at document boundaries
        if actual_length is not None and actual_length < x.shape[1]:
            # Only compute loss on valid (non-padded) tokens
            # actual_length-1 because we predict next token
            loss = F.cross_entropy(
                logits[:, :actual_length-1].transpose(1, 2),
                labels[:, :actual_length-1]
            )
        else:
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
            
            # Initialize conv block (two-stage like minGRU/FF)
            conv_block = layer[0]  # CausalConv1d if it exists
            if conv_block is not None:
                # Stage 1: Conv gets SCALED initialization
                nn.init.kaiming_normal_(conv_block.conv.weight, mode='fan_in', nonlinearity='linear')
                conv_block.conv.weight.data *= 0.1  # Scale down conv outputs
                
                # Stage 2: Projection gets SMALL random initialization  
                # Small enough to be near identity, but non-zero for gradient flow
                nn.init.normal_(conv_block.proj.weight, mean=0.0, std=0.01)
                # This makes initial behavior: output ≈ small, so x + small ≈ x (near identity)

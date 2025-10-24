import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from mingru.minGRU import minGRU

# Import streaming loss kernel
try:
    from mingru.triton_streaming_loss import triton_streaming_cross_entropy
    STREAMING_LOSS_AVAILABLE = True
    print("Triton streaming loss available (NO logits materialization!)")
except ImportError as e:
    print(f"Triton streaming loss not available: {e}")
    STREAMING_LOSS_AVAILABLE = False

# Import GRU implementations
try:
    from mingru.hybrid_fused_gru import HybridFusedGRU
    print("HybridFusedGRU available (PyTorch matmul + Triton fused cell)")
except ImportError as e:
    print(f"Failed to import hybrid_fused_gru: {e}")
    HybridFusedGRU = None

# Import test GRU for debugging
try:
    from mingru.test_gru import TestGRU
    print("Test GRU available")
except ImportError:
    TestGRU = None

# Import standard GRU (cuDNN-optimized)
try:
    from mingru.standard_gru import StandardGRU
    print("StandardGRU available (cuDNN-optimized)")
except ImportError as e:
    print(f"Failed to import StandardGRU: {e}")
    StandardGRU = None

# Import lightweight causal conv GRU (NO cuDNN!)
try:
    from mingru.causal_conv_gru import CausalConvGRU
    print("CausalConvGRU available (lightweight, NO cuDNN bloat!)")
except ImportError as e:
    print(f"Failed to import CausalConvGRU: {e}")
    CausalConvGRU = None

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
        use_hybrid_gru = False,  # Use HybridFusedGRU with Triton kernel
        use_test_gru = False,  # Use simple test GRU
        use_standard_gru = False,  # Use PyTorch nn.GRU (cuDNN, gold standard)
        use_causal_conv_gru = False,  # Use lightweight causal conv (NO cuDNN!)
        z_bias_input = -2.0,  # Initial bias for z-gates on input projection
        z_bias_hidden = -2.0  # Initial bias for z-gates on hidden projection
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

        # Choose RNN class based on flags
        if use_test_gru:
            min_rnn_klass = TestGRU
            print(f"Using Test GRU for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_causal_conv_gru:
            min_rnn_klass = CausalConvGRU
            print(f"Using CausalConvGRU (lightweight causal conv, NO cuDNN bloat!) for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_standard_gru:
            min_rnn_klass = StandardGRU
            print(f"Using StandardGRU (cuDNN-optimized, gold standard) for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_hybrid_gru:
            min_rnn_klass = HybridFusedGRU
            # HybridFusedGRU uses PyTorch matmul + Triton fused cell
            print(f"Using HybridFusedGRU (Triton kernel) for depth={depth} model")

            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_hybrid_gru': use_hybrid_gru,
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden
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
        
        # Store configuration for checkpointing and generation
        self.dim = dim
        self.depth = depth
        self.use_hybrid_gru = use_hybrid_gru
        self.use_test_gru = use_test_gru
        
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

            # Handle both 2-value and 3-value returns (conv_buffers optional)
            mingru_result = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden = True
            )

            # Unpack result (handles 2 or 3 return values)
            if isinstance(mingru_result, tuple) and len(mingru_result) == 3:
                min_gru_out, next_prev_hidden, _ = mingru_result  # Ignore conv_buffers from StandardGRU
            else:
                min_gru_out, next_prev_hidden = mingru_result

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward

            if exists(ff) and exists(ff_norm):
                x = ff(ff_norm(x)) + x
            
            # dropout
            
            if exists(dropout):
                x = dropout(x)

        embed = self.norm(x)

        if not return_loss:
            # Inference: materialize full logits (needed for generation)
            logits = self.to_logits(embed)
            if not return_prev_hiddens:
                return logits
            # Return both RNN hiddens and conv buffers for inference
            return logits, (next_prev_hiddens, next_conv_buffers)

        # TRAINING: Chunked loss calculation - process 512 tokens at once
        # This is 4× faster than 64-position batches while still saving ~85% memory
        # Full logits would be [batch, seq, vocab] = batch × 2048 × 100K × 4 bytes
        # Chunked is [batch, 512, vocab] = batch × 512 × 100K × 4 bytes (4× smaller)

        # Vectorized loss masking
        labels_masked = labels.clone()
        if actual_length is not None and torch.is_tensor(actual_length):
            seq_len = labels.size(1)
            arange = torch.arange(seq_len, device=labels.device)[None, :]
            mask = arange >= (actual_length - 1)[:, None]
            labels_masked[mask] = -100

        # Use Triton streaming loss (NO logits materialization!)
        # NOTE: Triton kernel has bugs (nested loops), using chunked loss instead
        # Chunked loss: minimize logits materialization to push batch_size to max!
        if False and STREAMING_LOSS_AVAILABLE:
            loss = triton_streaming_cross_entropy(embed, self.to_logits.weight, labels_masked)
        else:
            # Chunked loss: MINIMAL chunks to reduce memory, enable larger batch_size!
            seq_len = embed.size(1)
            chunk_size = 16  # 16 tokens = quarter the logits memory vs 64!
            total_loss = 0.0
            num_valid = 0

            for i in range(0, seq_len, chunk_size):
                end = min(i + chunk_size, seq_len)
                logits_chunk = self.to_logits(embed[:, i:end])
                labels_chunk = labels_masked[:, i:end]

                valid_mask = labels_chunk != -100
                valid_count = valid_mask.sum().item()

                if valid_count > 0:
                    loss_chunk = F.cross_entropy(
                        logits_chunk.transpose(1, 2),
                        labels_chunk,
                        ignore_index=-100,
                        reduction='sum'
                    )
                    total_loss += loss_chunk
                    num_valid += valid_count

            loss = total_loss / max(num_valid, 1)

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

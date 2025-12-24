import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from mingru.minGRU import minGRU

# Import streaming loss kernel (FIXED VERSION!)
try:
    from mingru.triton_streaming_loss_fixed import triton_streaming_cross_entropy
    STREAMING_LOSS_AVAILABLE = True
    print("Triton streaming loss available (FIXED - NO logits materialization!)")
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

# Import Parallel EMA GRU (10x faster than sequential version!)
try:
    from mingru.parallel_ema_gru import ParallelEMA_GRU as HybridFusedGRU_EMA
    print("ParallelEMA_GRU available (10x faster: parallel EMA + Triton GRU)")
except ImportError as e:
    # Fallback to sequential version
    try:
        from mingru.hybrid_fused_gru_ema import HybridFusedGRU_EMA
        print("HybridFusedGRU_EMA available (sequential, slower)")
    except ImportError:
        print(f"Failed to import EMA GRU: {e}")
        HybridFusedGRU_EMA = None

# Import FlashGRU_EMA (FlashRNN + parallel EMA - 60x faster!)
try:
    from mingru.flash_gru_ema import FlashGRU_EMA
    print("FlashGRU_EMA available (FlashRNN + parallel EMA, 60x faster!)")
except ImportError as e:
    print(f"FlashGRU_EMA not available: {e}")
    FlashGRU_EMA = None

# Import CuDNNGRU_EMA (cuDNN GRU + parallel EMA - DDP-compatible!)
try:
    from mingru.cudnn_gru_ema import CuDNNGRU_EMA, CuDNNGRU_MultiScaleEMA
    print("CuDNNGRU_EMA available (cuDNN GRU + parallel EMA, DDP-compatible!)")
except ImportError as e:
    print(f"CuDNNGRU_EMA not available: {e}")
    CuDNNGRU_EMA = None
    CuDNNGRU_MultiScaleEMA = None

# Import CuDNNGRU_SSM (cuDNN GRU + Selective SSM - advanced memory!)
try:
    from mingru.cudnn_gru_ssm import CuDNNGRU_SSM
    print("CuDNNGRU_SSM available (cuDNN GRU + Selective SSM, DDP-compatible!)")
except ImportError as e:
    print(f"CuDNNGRU_SSM not available: {e}")
    CuDNNGRU_SSM = None

# Import CuDNNGRU_SSM_Series (cuDNN GRU → Selective SSM in series!)
try:
    from mingru.cudnn_gru_ssm_series import CuDNNGRU_SSM_Series
    print("CuDNNGRU_SSM_Series available (GRU→SSM series, DDP-compatible!)")
except ImportError as e:
    print(f"CuDNNGRU_SSM_Series not available: {e}")
    CuDNNGRU_SSM_Series = None

# Import cuDNN Fused GRU (3× faster than Hybrid!)
try:
    from mingru.cudnn_fused_gru import CuDNNFusedGRU
    print("CuDNNFusedGRU available (cuDNN kernel, 3× faster than Hybrid!)")
except ImportError as e:
    print(f"Failed to import CuDNNFusedGRU: {e}")
    CuDNNFusedGRU = None

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

# Import persistent GRU (optimized persistent-T kernel)
try:
    from mingru.persistent_gru import PersistentGRU
    print("PersistentGRU available (optimized persistent-T kernel)")
except ImportError as e:
    print(f"Failed to import PersistentGRU: {e}")
    PersistentGRU = None

# Import sequential Triton GRU (non-persistent, eliminates data race)
try:
    from mingru.triton_gru_sequential import SequentialTritonGRU
    print("SequentialTritonGRU available (non-persistent, no data race)")
except ImportError as e:
    print(f"Failed to import SequentialTritonGRU: {e}")
    SequentialTritonGRU = None

# Import selective GRU (input-dependent gating, inspired by Mamba)
try:
    from mingru.selective_gru import SelectiveGRU
    print("SelectiveGRU available (input-dependent Δ and selection)")
except ImportError as e:
    print(f"Failed to import SelectiveGRU: {e}")
    SelectiveGRU = None

# Import projected GRU (reduced recurrent dimension for 10% speedup + 67% larger batch)
try:
    from mingru.projected_gru import ProjectedGRU
    print("ProjectedGRU available (H_rec<D optimization, 10% faster + 40% fewer params)")
except ImportError as e:
    print(f"Failed to import ProjectedGRU: {e}")
    ProjectedGRU = None

# Import local conv window (NOT recurrent - limited receptive field!)
try:
    from mingru.local_conv_gru import LocalConvGRU
    print("LocalConvGRU available (local conv window, NOT recurrent!)")
except ImportError as e:
    print(f"Failed to import LocalConvGRU: {e}")
    LocalConvGRU = None

# Import FlashRNN GRU (hardware-optimized, 50x faster!)
try:
    from mingru.flash_gru import FlashGRU
    print("FlashGRU available (FlashRNN optimized, 50x speedup!)")
except ImportError as e:
    print(f"Failed to import FlashGRU: {e}")
    FlashGRU = None

# Import ElmanSilu (haste CUDA kernels, 3x faster than cuDNN GRU!)
try:
    from mingru.elman_silu import ElmanSilu
    print("ElmanSilu available (haste CUDA kernels, 3x faster than cuDNN GRU!)")
except ImportError as e:
    print(f"Failed to import ElmanSilu: {e}")
    ElmanSilu = None

# Import ElmanLeaky (true discretized Elman with input-dependent delta, Mamba2-style)
try:
    from mingru.elman_leaky import ElmanLeaky
    print("ElmanLeaky available (true discretized dynamics, input-dependent delta!)")
except ImportError as e:
    print(f"Failed to import ElmanLeaky: {e}")
    ElmanLeaky = None

# Import ElmanLeakySelective (Mamba2-style discretization + h+x output gate)
try:
    from mingru.elman_leaky_selective import ElmanLeakySelective
    print("ElmanLeakySelective available (Mamba2-style discretization + h+x output gate!)")
except ImportError as e:
    print(f"Failed to import ElmanLeakySelective: {e}")
    ElmanLeakySelective = None

# Import LeakyElman (leaky integration + input-only output gate)
try:
    from mingru.leaky_elman import LeakyElman
    print("LeakyElman available (leaky integration + INPUT-ONLY output gate!)")
except ImportError as e:
    print(f"Failed to import LeakyElman: {e}")
    LeakyElman = None

# Import HasteGRUSilu (haste GRU + silu output gate, matches cuDNN GRU + silu!)
try:
    from mingru.haste_gru_silu import HasteGRUSilu
    print("HasteGRUSilu available (haste GRU + silu gate, proper skip connection!)")
except ImportError as e:
    print(f"Failed to import HasteGRUSilu: {e}")
    HasteGRUSilu = None

# Import HasteGRUSiluFused (fused GRU + silu CUDA kernel, BF16 native!)
try:
    from mingru.haste_gru_silu_fused import HasteGRUSiluFused
    print("HasteGRUSiluFused available (fused CUDA kernel, BF16 native!)")
except ImportError as e:
    print(f"Failed to import HasteGRUSiluFused: {e}")
    HasteGRUSiluFused = None

# Import HasteLSTMSilu (fused LSTM + silu CUDA kernel, BF16 native!)
try:
    from mingru.haste_lstm_silu import HasteLSTMSilu
    print("HasteLSTMSilu available (fused LSTM + silu CUDA kernel, BF16 native!)")
except ImportError as e:
    print(f"Failed to import HasteLSTMSilu: {e}")
    HasteLSTMSilu = None

# Import SkipElmanSilu (SkipElman + silu output gate, simpler than GRU!)
try:
    from mingru.skip_elman_silu import SkipElmanSilu
    print("SkipElmanSilu available (SkipElman + silu gate, gradient highway!)")
except ImportError as e:
    print(f"Failed to import SkipElmanSilu: {e}")
    SkipElmanSilu = None

# Import ElmanSwish (silu everywhere - like Mamba2 internal activations!)
try:
    from mingru.elman_swish import ElmanSwishLayer
    print("ElmanSwishLayer available (silu inside + silu gate, SwiGLU-style!)")
except ImportError as e:
    print(f"Failed to import ElmanSwishLayer: {e}")
    ElmanSwishLayer = None

# Import ElmanInputGate (input-only gating like Mamba2!)
try:
    from mingru.elman_input_gate import ElmanInputGate
    print("ElmanInputGate available (input-only gating like Mamba2!)")
except ImportError as e:
    print(f"Failed to import ElmanInputGate: {e}")
    ElmanInputGate = None

# Import MultiHeadElman (multi-head RNN with per-head R matrices, 2048x more expressive than Mamba2!)
try:
    from mingru.multihead_elman import MultiHeadElman
    print("MultiHeadElman available (32 heads × 64×64 R matrices, 2048x more expressive than Mamba2!)")
except ImportError as e:
    print(f"Failed to import MultiHeadElman: {e}")
    MultiHeadElman = None

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
        use_fused_gru = False,  # Use Fused GRU (cuDNN kernel, 3× faster)
        use_hybrid_gru = False,  # Backwards compat alias for use_fused_gru
        use_test_gru = False,  # Use simple test GRU
        use_standard_gru = False,  # Use PyTorch nn.GRU (cuDNN, gold standard)
        use_persistent_gru = False,  # Use PersistentGRU (optimized persistent-T kernel)
        use_sequential_triton_gru = False,  # Use SequentialTritonGRU (non-persistent, no data race)
        use_selective_gru = False,  # Use SelectiveGRU (input-dependent Δ and selection)
        use_projected_gru = False,  # Use ProjectedGRU (10% faster + 67% larger batch!)
        h_recurrent = None,  # Recurrent dimension for ProjectedGRU (default: dim*0.625)
        use_local_conv = False,  # Use LocalConvGRU (local window, NOT recurrent!)
        use_flash_gru = False,  # Use FlashRNN GRU (50x faster, hardware-optimized!)
        use_flash_ema_gru = False,  # Use FlashRNN GRU + EMA (60x faster + long-range memory!)
        use_cudnn_ema_gru = False,  # Use cuDNN GRU + EMA (DDP-compatible + long-range memory!)
        use_cudnn_multiscale_ema_gru = False,  # Use cuDNN GRU + Multi-Scale EMA (3 timescales!)
        use_cudnn_ssm_gru = False,  # Use cuDNN GRU + Selective SSM (learned decay, input-dependent!)
        use_cudnn_ssm_series_gru = False,  # Use cuDNN GRU → SSM in series (GRU extracts, SSM tracks!)
        per_layer_alpha = False,  # Initialize each layer with different EMA alpha (fast→slow)
        use_ema_gru = False,  # Use EMA GRU (GRU + EMA for long-range memory)
        use_elman_silu = False,  # Use ElmanSilu (haste CUDA, 3x faster than cuDNN GRU!)
        use_elman_leaky = False,  # Use ElmanLeaky (true discretized dynamics, input-dependent delta!)
        use_elman_leaky_selective = False,  # Use ElmanLeakySelective (Mamba2-style discretization + h+x output gate!)
        use_leaky_elman = False,  # Use LeakyElman (leaky integration + INPUT-ONLY output gate!)
        delta_init = -2.0,  # Delta initialization for ElmanLeaky/ElmanLeakySelective/ElmanMamba
        use_haste_gru_silu = False,  # Use HasteGRUSilu (haste GRU + silu, proper skip connection!)
        use_haste_gru_silu_fused = False,  # Use HasteGRUSiluFused (fused CUDA kernel, BF16 native!)
        use_haste_lstm_silu = False,  # Use HasteLSTMSilu (fused LSTM + silu CUDA kernel, BF16 native!)
        use_skip_elman_silu = False,  # Use SkipElmanSilu (SkipElman + silu, simpler than GRU!)
        use_elman_swish = False,  # Use ElmanSwish (silu inside + silu gate, like Mamba2!)
        use_elman_input_gate = False,  # Use ElmanInputGate (input-only gating like Mamba2!)
        use_multihead_elman = False,  # Use MultiHeadElman (32 heads × 64×64 R matrices!)
        multihead_elman_nheads = 32,  # Number of heads for MultiHeadElman
        multihead_elman_headdim = 64,  # Dimension per head for MultiHeadElman
        multihead_elman_activation = 'softsign',  # Activation: 'softsign' or 'tanh_residual'
        ema_alpha = 0.01,  # EMA decay rate (small = longer memory, 0.01 ~ 70 token half-life)
        use_gradient_checkpointing = False,  # Use gradient checkpointing to reduce memory
        z_bias_input = -2.0,  # Initial bias for z-gates on input projection
        z_bias_hidden = -2.0,  # Initial bias for z-gates on hidden projection
        recurrence_chunk_size = 64  # Chunk size for GRU recurrence (reduces kernel launches)
    ):
        super().__init__()
        
        # Handle backwards compatibility
        if enable_conv is not None:
            # Old style parameter - convert to new style
            if enable_conv:
                conv_kernel_size = conv_kernel_size if conv_kernel_size != 3 else 3
            else:
                conv_kernel_size = None

        # Handle use_hybrid_gru as alias for use_fused_gru
        if use_hybrid_gru:
            use_fused_gru = True
        
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        # Choose RNN class based on flags
        if use_test_gru:
            min_rnn_klass = TestGRU
            print(f"Using Test GRU for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_local_conv:
            min_rnn_klass = LocalConvGRU
            print(f"Using LocalConvGRU (local conv window, NOT recurrent!) for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_flash_gru:
            min_rnn_klass = FlashGRU
            print(f"Using FlashGRU (FlashRNN optimized, 50x speedup!) for depth={depth} model")
            rnn_kwargs = {'expansion_factor': expansion}
        elif use_flash_ema_gru:
            # FlashRNN GRU + parallel EMA for fast recurrence with long-range memory
            if FlashGRU_EMA is None:
                raise ImportError("FlashGRU_EMA not available. Install flashrnn: pip install flashrnn")
            min_rnn_klass = FlashGRU_EMA
            half_life = int(0.693 / ema_alpha) if ema_alpha > 0 else float('inf')
            print(f"Using FlashGRU_EMA (FlashRNN + parallel EMA, 60x faster, half-life={half_life} tokens) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'ema_alpha': ema_alpha,
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_cudnn_ssm_series_gru:
            # cuDNN GRU → Selective SSM in series - GRU extracts features, SSM tracks state
            if CuDNNGRU_SSM_Series is None:
                raise ImportError("CuDNNGRU_SSM_Series not available")
            min_rnn_klass = CuDNNGRU_SSM_Series
            print(f"Using CuDNNGRU_SSM_Series (GRU→SSM series, DDP-compatible) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'selective': True,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_cudnn_ssm_gru:
            # cuDNN GRU + Selective SSM - advanced memory with learned decay
            if CuDNNGRU_SSM is None:
                raise ImportError("CuDNNGRU_SSM not available")
            min_rnn_klass = CuDNNGRU_SSM
            print(f"Using CuDNNGRU_SSM (cuDNN + Selective SSM, learned decay, DDP-compatible) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'selective': True,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_cudnn_multiscale_ema_gru:
            # cuDNN GRU + multi-scale parallel EMA - captures multiple timescales
            if CuDNNGRU_MultiScaleEMA is None:
                raise ImportError("CuDNNGRU_MultiScaleEMA not available")
            min_rnn_klass = CuDNNGRU_MultiScaleEMA
            print(f"Using CuDNNGRU_MultiScaleEMA (cuDNN + 3-scale EMA: 0.1/0.01/0.001, DDP-compatible) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'num_scales': 3,
                'init_alphas': (0.1, 0.01, 0.001),
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden,
                'recurrence_chunk_size': recurrence_chunk_size
            }
            # Always pass layer info for multi-scale (beneficial for depth-aware timescales)
            rnn_kwargs['_per_layer'] = True
        elif use_cudnn_ema_gru:
            # cuDNN GRU + parallel EMA - DDP-compatible alternative to FlashGRU_EMA
            if CuDNNGRU_EMA is None:
                raise ImportError("CuDNNGRU_EMA not available")
            min_rnn_klass = CuDNNGRU_EMA
            half_life = int(0.693 / ema_alpha) if ema_alpha > 0 else float('inf')
            if per_layer_alpha:
                print(f"Using CuDNNGRU_EMA with per-layer alpha (layer 0: α≈0.05 → layer {depth-1}: α≈0.005) for depth={depth} model")
            else:
                print(f"Using CuDNNGRU_EMA (cuDNN + parallel EMA, DDP-compatible, half-life={half_life} tokens) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'ema_alpha': ema_alpha,
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden,
                'recurrence_chunk_size': recurrence_chunk_size
            }
            if per_layer_alpha:
                rnn_kwargs['_per_layer'] = True
        elif use_standard_gru:
            min_rnn_klass = StandardGRU
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            # Combine z_bias_input and z_bias_hidden (nn.GRU has single gate bias)
            z_bias_combined = (z_bias_input + z_bias_hidden) / 2
            print(f"Using StandardGRU (cuDNN-optimized{checkpoint_str}, chunk_size={recurrence_chunk_size}, z_bias={z_bias_combined}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size,
                'z_bias_init': z_bias_combined
            }
        elif use_persistent_gru:
            min_rnn_klass = PersistentGRU
            print(f"Using PersistentGRU (optimized persistent-T kernel) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion
            }
        elif use_sequential_triton_gru:
            min_rnn_klass = SequentialTritonGRU
            print(f"Using SequentialTritonGRU (non-persistent, no data race) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion
            }
        elif use_selective_gru:
            min_rnn_klass = SelectiveGRU
            print(f"Using SelectiveGRU (input-dependent Δ, selection) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion
            }
        elif use_projected_gru:
            min_rnn_klass = ProjectedGRU
            # Default H_rec = 0.625 * dim (optimal from benchmarks: 1280 for dim=2048)
            h_rec_actual = h_recurrent if h_recurrent is not None else int(dim * 0.625)
            print(f"Using ProjectedGRU (H_rec={h_rec_actual}, 10% faster + 67% larger batch!) for depth={depth} model")
            rnn_kwargs = {
                'h_recurrent': h_rec_actual,
                'expansion_factor': expansion
            }
        elif use_ema_gru:
            # Use ParallelEMA_GRU (10x faster: parallel EMA + Triton GRU cell)
            min_rnn_klass = HybridFusedGRU_EMA
            half_life = int(0.693 / ema_alpha) if ema_alpha > 0 else float('inf')
            print(f"Using ParallelEMA_GRU (10x faster EMA + Triton GRU, alpha={ema_alpha}, half-life={half_life} tokens) for depth={depth} model")

            rnn_kwargs = {
                'expansion_factor': expansion,
                'ema_alpha': ema_alpha,
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_fused_gru:
            # Use HybridFusedGRU (Triton fused kernel)
            # Note: CuDNNFusedGRU is 3× faster but materializes all timesteps → OOM for large models
            min_rnn_klass = HybridFusedGRU
            print(f"Using HybridFusedGRU (Triton fused kernel, chunk_size={recurrence_chunk_size}) for depth={depth} model")

            rnn_kwargs = {
                'expansion_factor': expansion,
                'z_bias_input': z_bias_input,
                'z_bias_hidden': z_bias_hidden,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_elman_silu:
            # Use ElmanSilu (haste CUDA kernels, 3x faster than cuDNN GRU!)
            min_rnn_klass = ElmanSilu
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using ElmanSilu (haste CUDA{checkpoint_str}, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_elman_leaky:
            # Use ElmanLeaky (true discretized dynamics, input-dependent delta!)
            min_rnn_klass = ElmanLeaky
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using ElmanLeaky (true discretized{checkpoint_str}, delta_init={delta_init}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size,
                'delta_init': delta_init
            }
        elif use_elman_leaky_selective:
            # Use ElmanLeakySelective (Mamba2-style discretization + h+x output gate!)
            min_rnn_klass = ElmanLeakySelective
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using ElmanLeakySelective (Mamba2-style{checkpoint_str}, delta_init={delta_init}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size,
                'delta_init': delta_init
            }
        elif use_leaky_elman:
            # Use LeakyElman (leaky integration + input-only output gate)
            min_rnn_klass = LeakyElman
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using LeakyElman (leaky integration + INPUT-ONLY gate{checkpoint_str}, delta_init={delta_init}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size,
                'delta_init': delta_init,
                'use_output_gate': True  # Input-only gate
            }
        elif use_haste_gru_silu:
            # Use HasteGRUSilu (haste GRU + silu output gate, proper skip connection!)
            min_rnn_klass = HasteGRUSilu
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using HasteGRUSilu (haste GRU + silu gate{checkpoint_str}, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_haste_gru_silu_fused:
            # Use HasteGRUSiluFused (fused CUDA kernel, BF16 native!)
            min_rnn_klass = HasteGRUSiluFused
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using HasteGRUSiluFused (fused CUDA{checkpoint_str}, BF16 native, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_haste_lstm_silu:
            # Use HasteLSTMSilu (fused LSTM + silu CUDA kernel, BF16 native!)
            min_rnn_klass = HasteLSTMSilu
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using HasteLSTMSilu (fused LSTM+silu{checkpoint_str}, BF16 native, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_skip_elman_silu:
            # Use SkipElmanSilu (SkipElman + silu output gate, simpler than GRU!)
            min_rnn_klass = SkipElmanSilu
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using SkipElmanSilu (SkipElman + silu gate{checkpoint_str}, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_elman_swish:
            # Use ElmanSwish (silu inside + silu gate, like Mamba2 activations!)
            min_rnn_klass = ElmanSwishLayer
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using ElmanSwish (silu+silu{checkpoint_str}, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_elman_input_gate:
            # Use ElmanInputGate (input-only gating like Mamba2!)
            min_rnn_klass = ElmanInputGate
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            print(f"Using ElmanInputGate (input-only gate{checkpoint_str}, chunk_size={recurrence_chunk_size}) for depth={depth} model")
            rnn_kwargs = {
                'expansion_factor': expansion,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        elif use_multihead_elman:
            # Use MultiHeadElman (32 heads × 64×64 R matrices, 2048x more expressive than Mamba2!)
            min_rnn_klass = MultiHeadElman
            checkpoint_str = " with gradient checkpointing" if use_gradient_checkpointing else ""
            r_params = multihead_elman_nheads * multihead_elman_headdim * multihead_elman_headdim
            print(f"Using MultiHeadElman ({multihead_elman_nheads} heads × {multihead_elman_headdim}×{multihead_elman_headdim}, {r_params:,} R params, {multihead_elman_activation}{checkpoint_str}) for depth={depth} model")
            rnn_kwargs = {
                'nheads': multihead_elman_nheads,
                'headdim': multihead_elman_headdim,
                'expansion_factor': expansion,
                'activation': multihead_elman_activation,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'recurrence_chunk_size': recurrence_chunk_size
            }
        else:
            min_rnn_klass = minGRU
            rnn_kwargs = {'expansion_factor': expansion}

        # Check if per-layer alpha initialization is needed
        use_per_layer_init = rnn_kwargs.pop('_per_layer', False)

        for layer_idx in range(depth):
            # Add layer info if per-layer alpha is enabled
            layer_rnn_kwargs = rnn_kwargs.copy()
            if use_per_layer_init:
                layer_rnn_kwargs['layer_idx'] = layer_idx
                layer_rnn_kwargs['num_layers'] = depth

            self.layers.append(ModuleList([
                CausalConv1d(dim, conv_kernel_size) if conv_kernel_size else None,
                RMSNorm(dim),
                min_rnn_klass(dim, **layer_rnn_kwargs),
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
        actual_length = None,  # For masking padded chunks at doc boundaries
        doc_boundaries = None  # [B, T] boolean tensor: True = reset hidden state at this token
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
                return_next_prev_hidden = True,
                doc_boundaries = doc_boundaries
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
        # NOTE: Kernel works but needs custom autograd.Function for gradients
        # TODO: Implement backward pass, for now use chunked fallback
        if False and STREAMING_LOSS_AVAILABLE:
            loss = triton_streaming_cross_entropy(embed, self.to_logits.weight, labels_masked)
        else:
            # Chunked loss: Balance speed and memory for profiling baseline
            seq_len = embed.size(1)
            chunk_size = 64  # OPTIMAL: Tested 16,32,64,128 - 64 is fastest!
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

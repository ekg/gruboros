"""
Hybrid GRU: PyTorch matmuls + Triton fused cell logic.

Best of both worlds:
- PyTorch's optimized CUBLAS for matrix multiplication
- Triton kernel for fused GRU cell computation
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


def gru_cell_pytorch(input_gates, hidden_gates, h_prev):
    """Pure PyTorch GRU cell for CPU/debugging."""
    B, H3 = input_gates.shape
    H = H3 // 3
    
    # Split gates
    i_r, i_z, i_n = input_gates.chunk(3, dim=-1)
    h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)
    
    # GRU computation
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h_new = (1 - z) * h_prev + z * n
    
    return h_new


@triton.jit
def gru_chunk_fused_with_doc_boundaries(
    # Input gates [B, chunk_size, 3*H]
    input_gates_ptr,
    # Weight matrix [H, 3*H] for hidden projection
    weight_ptr,
    # Initial hidden state [B, H]
    h_in_ptr,
    # Document boundaries [B, chunk_size] as int8 (0 or 1)
    doc_boundaries_ptr,
    # Outputs: all hidden states [B, chunk_size, H]
    outputs_ptr,
    # Final hidden state [B, H]
    h_out_ptr,
    # Dimensions
    batch_size, hidden_dim, chunk_size,
    # Weight strides
    weight_stride_0, weight_stride_1,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Chunked GRU with matmul INSIDE kernel + document boundary resets.

    Key: Process chunk_size timesteps in ONE kernel launch!
    For each timestep sequentially:
    - Compute h_gates = h @ W (current h in registers)
    - Apply GRU cell
    - Check document boundary and reset h if needed
    - Update h for next iteration

    Reduces 512 launches → 8 launches (64× reduction!)
    """
    pid_batch = tl.program_id(0)
    if pid_batch >= batch_size:
        return

    offs_h = tl.arange(0, BLOCK_SIZE)
    mask_h = offs_h < hidden_dim

    # Load initial h [H] in fp32
    h_base = pid_batch * hidden_dim
    h = tl.load(h_in_ptr + h_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    # Process timesteps sequentially INSIDE this kernel!
    for t in range(chunk_size):
        # === Compute hidden_gates = h @ W (matvec) ===
        # For each of 3 output vectors (r,z,n), compute dot product with h

        # Initialize gate accumulators
        h_r = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        h_z = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        h_n = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Accumulate: for each output element j, sum_i h[i] * W[i,j]
        # Load h values with proper masking
        h_masked = tl.where(mask_h, h, 0.0)

        # Compute outer product and sum
        for i in range(BLOCK_SIZE):
            # Get scalar h[i]
            h_i = h_masked[i]

            # Load weight row i for all three gates
            w_r_row = tl.load(weight_ptr + i * weight_stride_0 + offs_h * weight_stride_1,
                             mask=(i < hidden_dim) & mask_h, other=0.0).to(tl.float32)
            h_r += h_i * w_r_row

            w_z_row = tl.load(weight_ptr + i * weight_stride_0 + (hidden_dim + offs_h) * weight_stride_1,
                             mask=(i < hidden_dim) & mask_h, other=0.0).to(tl.float32)
            h_z += h_i * w_z_row

            w_n_row = tl.load(weight_ptr + i * weight_stride_0 + (2*hidden_dim + offs_h) * weight_stride_1,
                             mask=(i < hidden_dim) & mask_h, other=0.0).to(tl.float32)
            h_n += h_i * w_n_row

        # === Load input gates for this timestep ===
        gates_base = pid_batch * chunk_size * 3 * hidden_dim + t * 3 * hidden_dim
        i_r = tl.load(input_gates_ptr + gates_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        i_z = tl.load(input_gates_ptr + gates_base + hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        i_n = tl.load(input_gates_ptr + gates_base + 2*hidden_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        # === GRU cell ===
        r = tl.sigmoid(i_r + h_r)
        z = tl.sigmoid(i_z + h_z)

        n_pre = i_n + r * h_n
        n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
        exp_2x = tl.exp(2.0 * n_pre_clamped)
        n = (exp_2x - 1.0) / (exp_2x + 1.0)

        h = (1.0 - z) * h + z * n  # Update h for next iteration!

        # === Document boundary reset ===
        # Load document boundary flag for this batch element and timestep
        doc_bound_offset = pid_batch * chunk_size + t
        doc_boundary = tl.load(doc_boundaries_ptr + doc_bound_offset)  # scalar: 0 or 1

        # Reset h to zero if this is a document boundary
        # Use where to conditionally zero out h
        h = tl.where(doc_boundary > 0.5, 0.0, h)

        # Write output
        output_offset = pid_batch * chunk_size * hidden_dim + t * hidden_dim + offs_h
        tl.store(outputs_ptr + output_offset, h, mask=mask_h)

    # Write final h
    tl.store(h_out_ptr + h_base + offs_h, h, mask=mask_h)


@triton.jit
def gru_cell_fused(
    # Gate inputs from matmul
    gates_input_ptr, gates_hidden_ptr,
    # Hidden state
    h_in_ptr, h_out_ptr,
    # Dimensions
    batch_size, hidden_dim,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused GRU cell computation ONLY.
    Assumes gates are already computed by PyTorch matmul.

    Input: gates_input[B, 3*H], gates_hidden[B, 3*H], h_in[B, H]
    Output: h_out[B, H]
    """
    # Program for each batch element
    pid_batch = tl.program_id(0)
    
    if pid_batch >= batch_size:
        return
    
    # Since we process the entire hidden dimension in one block,
    # we don't need block-level partitioning
    # Note: BLOCK_SIZE must be a power of 2 for tl.arange
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    
    # Load input gates for this block (convert to fp32 for computation)
    base_offset = pid_batch * 3 * hidden_dim
    
    i_r = tl.load(gates_input_ptr + base_offset + offs, mask=mask, other=0.0).to(tl.float32)
    i_z = tl.load(gates_input_ptr + base_offset + hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    i_n = tl.load(gates_input_ptr + base_offset + 2*hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Load hidden gates for this block (convert to fp32 for computation)
    h_r = tl.load(gates_hidden_ptr + base_offset + offs, mask=mask, other=0.0).to(tl.float32)
    h_z = tl.load(gates_hidden_ptr + base_offset + hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    h_n = tl.load(gates_hidden_ptr + base_offset + 2*hidden_dim + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Load previous hidden state (convert to fp32 for computation)
    h_prev_offset = pid_batch * hidden_dim + offs
    h_prev = tl.load(h_in_ptr + h_prev_offset, mask=mask, other=0.0).to(tl.float32)
    
    # GRU cell computation (fused)
    r = tl.sigmoid(i_r + h_r)
    z = tl.sigmoid(i_z + h_z)
    
    # Candidate - numerically safer tanh using available Triton functions
    n_pre = i_n + r * h_n
    # Clamp input to avoid overflow (tanh saturates at ±3)
    n_pre_clamped = tl.minimum(tl.maximum(n_pre, -3.0), 3.0)
    exp_2x = tl.exp(2.0 * n_pre_clamped)
    n = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Update hidden
    h_new = (1.0 - z) * h_prev + z * n
    
    # Store result in fp32
    tl.store(h_out_ptr + h_prev_offset, h_new, mask=mask)


class HybridFusedGRU(nn.Module):
    """
    Hybrid GRU using PyTorch matmul + Triton cell fusion.
    
    This should be faster than pure Triton but with fusion benefits.
    """
    
    def __init__(self, dim: int, expansion_factor: float = 1.5, z_bias_input: float = -2.0, z_bias_hidden: float = -2.0, recurrence_chunk_size: int = 64, use_identity_init: bool = False, **kwargs):
        super().__init__()
        self.recurrence_chunk_size = recurrence_chunk_size
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.z_bias_input = z_bias_input
        self.z_bias_hidden = z_bias_hidden
        self.use_identity_init = use_identity_init
        
        # Standard GRU weights
        self.input_projection = nn.Linear(dim, 3 * self.dim_inner)
        self.hidden_projection = nn.Linear(self.dim_inner, 3 * self.dim_inner)
        
        # Only add output projection if expanding
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        import math
        std = 1.0 / math.sqrt(self.dim_inner)
        
        # Weights: mild uniform is fine
        for lin in [self.input_projection, self.hidden_projection]:
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)  # start from zeros
            
            # gates layout: [r | z | n] each of size H = dim_inner
            H = self.dim_inner
            with torch.no_grad():
                # Apply z-gate biases based on whether this is input or hidden projection
                if lin == self.input_projection:
                    lin.bias[H:2*H].fill_(self.z_bias_input)  # bias the update gate z on input
                else:  # hidden_projection
                    lin.bias[H:2*H].fill_(self.z_bias_hidden)  # bias the update gate z on hidden
        
        # Residual projection initialization
        if not isinstance(self.to_out, nn.Identity):
            if self.use_identity_init:
                # Small random init for deep networks (NOT identity - too strong with residuals!)
                # Identity would create passthrough, amplifying signal through deep residual stack
                # Use small init scaled for deep networks
                nn.init.normal_(self.to_out.weight, mean=0.0, std=0.01)
            else:
                # Zero init (default, relies on residuals for gradient flow)
                nn.init.constant_(self.to_out.weight, 0.0)
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, doc_boundaries=None):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        if prev_hidden is None:
            h = torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        else:
            # Squeeze any extra dimensions if needed
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h = prev_hidden if prev_hidden.shape[0] == B else torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
        
        # Special fast path for single-token generation (T=1)
        if T == 1:
            # Direct computation for single timestep
            input_gates = self.input_projection(x.squeeze(1)).contiguous()  # [B, 3*H]
            hidden_gates = self.hidden_projection(h).contiguous()  # [B, 3*H]
            
            if device.type == 'cuda':
                # Allocate fp32 buffer for Triton kernel output
                h_new_fp32 = torch.empty(B, self.dim_inner, device=device, dtype=torch.float32)
                
                # Launch Triton kernel - use next power of 2 as block size
                # Find next power of 2 >= dim_inner (required for tl.arange)
                import math
                BLOCK_SIZE = 2 ** math.ceil(math.log2(self.dim_inner))
                grid = (B,)  # Just one block per batch element
                
                gru_cell_fused[grid](
                    input_gates, hidden_gates,
                    h, h_new_fp32,
                    B, self.dim_inner,
                    BLOCK_SIZE
                )
                # Cast back to original dtype
                h_new = h_new_fp32.to(dtype)
            else:
                # CPU fallback
                h_new = gru_cell_pytorch(input_gates, hidden_gates, h)
            
            out = self.to_out(h_new.unsqueeze(1)) + x  # [B, 1, D]
            
            if return_next_prev_hidden:
                return out, h_new
            return out
        
        # Multi-timestep processing (for training and prompt processing)
        # Pre-compute ALL input projections at once (FAST!)
        input_gates_all = self.input_projection(x)  # [B, T, 3*H]

        outputs = []

        # Process in chunks to reduce kernel launches
        import math
        BLOCK_SIZE = 2 ** math.ceil(math.log2(self.dim_inner))
        chunk_size = self.recurrence_chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size  # Ceiling division

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, T)
            actual_chunk_size = chunk_end - chunk_start

            # Use chunked kernel for full chunks on CUDA
            # DISABLED: Chunked kernel has Triton compilation issues with dynamic indexing
            if False and device.type == 'cuda' and actual_chunk_size == chunk_size and chunk_size > 1:
                # Extract input gates for this chunk
                input_gates_chunk = input_gates_all[:, chunk_start:chunk_end].contiguous()  # [B, chunk_size, 3*H]

                # Allocate output tensors
                chunk_outputs = torch.empty(B, chunk_size, self.dim_inner, device=device, dtype=torch.float32)
                h_out = torch.empty(B, self.dim_inner, device=device, dtype=torch.float32)

                # Get weight matrix [3*H, H] → transpose to [H, 3*H]
                weight = self.hidden_projection.weight.t().contiguous()  # [H, 3*H]

                # Prepare document boundaries for this chunk [B, chunk_size] as float32
                if doc_boundaries is not None:
                    doc_boundaries_chunk = doc_boundaries[:, chunk_start:chunk_end].to(torch.float32).contiguous()
                else:
                    # No boundaries - all zeros
                    doc_boundaries_chunk = torch.zeros(B, chunk_size, device=device, dtype=torch.float32)

                # Launch chunked kernel - ONE launch for entire chunk!
                grid = (B,)
                gru_chunk_fused_with_doc_boundaries[grid](
                    input_gates_chunk,
                    weight,
                    h,
                    doc_boundaries_chunk,
                    chunk_outputs,
                    h_out,
                    B, self.dim_inner, chunk_size,
                    weight.stride(0), weight.stride(1),
                    BLOCK_SIZE
                )

                # Cast outputs back to original dtype
                chunk_outputs = chunk_outputs.to(dtype)
                h = h_out.to(dtype)

                # Append all outputs from chunk
                for t_offset in range(actual_chunk_size):
                    outputs.append(chunk_outputs[:, t_offset])

            else:
                # Fall back to per-timestep processing (CPU or partial chunks)
                for t in range(chunk_start, chunk_end):
                    # Get input gates for this timestep
                    input_gates = input_gates_all[:, t].contiguous()  # [B, 3*H]

                    # Compute hidden gates with PyTorch matmul
                    hidden_gates = self.hidden_projection(h).contiguous()  # [B, 3*H]

                    # Compute new hidden state
                    if device.type == 'cuda':
                        # Allocate fp32 buffer for Triton kernel output
                        h_new_fp32 = torch.empty(B, self.dim_inner, device=device, dtype=torch.float32)

                        # Launch Triton kernel for fused cell computation
                        grid = (B,)

                        gru_cell_fused[grid](
                            input_gates, hidden_gates,
                            h, h_new_fp32,
                            B, self.dim_inner,
                            BLOCK_SIZE
                        )
                        # Cast back to original dtype
                        h_new = h_new_fp32.to(dtype)
                    else:
                        # CPU fallback
                        h_new = gru_cell_pytorch(input_gates, hidden_gates, h)

                    h = h_new

                    # Reset hidden states in-place at document boundaries (NO allocation!)
                    if doc_boundaries is not None:
                        reset_mask = doc_boundaries[:, t]  # [B] - which batch elements reset at this timestep
                        if reset_mask.any():
                            h = h.masked_fill(reset_mask.unsqueeze(-1), 0.0)  # In-place zero-out

                    outputs.append(h)
        
        # Stack outputs and apply final projection
        h_seq = torch.stack(outputs, dim=1)
        out = self.to_out(h_seq) + x  # Residual
        
        if return_next_prev_hidden:
            return out, h
        return out


# Export with clearer name (keeping NAU_GRU for backwards compatibility)
NAU_GRU = HybridFusedGRU  # Deprecated alias
FusedGRU = HybridFusedGRU  # Preferred name


if __name__ == "__main__":
    print("Testing Hybrid Fused GRU...")
    
    device = 'cuda'
    B, T, D = 8, 64, 256
    
    model = HybridFusedGRU(D).to(device).to(torch.bfloat16)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16) * 0.1
    
    # Test forward
    out = model(x)
    print(f"✓ Forward pass successful!")
    print(f"Output shape: {out.shape}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens = B * T * 10
    print(f"Speed: {tokens/elapsed:.0f} tok/s")
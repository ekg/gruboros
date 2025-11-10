"""
cuDNN Fused GRU with Document Boundary Support

Uses PyTorch's nn.GRU (cuDNN backend) for maximum performance,
but adds support for document boundary hidden state resets.

Key advantages over HybridFusedGRU:
- Single fused cuDNN kernel per layer (vs T×2 kernel launches)
- No PyTorch/Triton kernel switching overhead
- Highly optimized by NVIDIA
- Handles document boundaries by chunking sequences
"""
import torch
import torch.nn as nn


class CuDNNFusedGRU(nn.Module):
    """
    Fully fused GRU using cuDNN with document boundary support.

    Strategy:
    - Use nn.GRU (cuDNN) for maximum performance
    - Handle document boundaries by processing sequence in chunks
    - Reset hidden state between documents

    This should be MUCH faster than HybridFusedGRU because:
    1. cuDNN uses a single fused kernel (no launch overhead)
    2. No PyTorch/Triton switching
    3. Heavily optimized by NVIDIA engineers
    """

    def __init__(self, dim: int, expansion_factor: float = 1.0,
                 z_bias_input: float = 0.0, z_bias_hidden: float = 0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.z_bias_input = z_bias_input
        self.z_bias_hidden = z_bias_hidden

        # Input projection
        self.input_projection = nn.Linear(dim, self.dim_inner, bias=False)

        # Core GRU (cuDNN backend)
        self.gru = nn.GRU(
            input_size=self.dim_inner,
            hidden_size=self.dim_inner,
            num_layers=1,
            batch_first=True,
            bias=True
        )

        # Output projection if expanding
        if expansion_factor != 1.0:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)
        else:
            self.to_out = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        import math
        std = 1.0 / math.sqrt(self.dim_inner)

        # Initialize input projection
        nn.init.uniform_(self.input_projection.weight, -std, std)

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.uniform_(param, -std, std)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Apply z-gate bias
                H = self.dim_inner
                if 'bias_ih' in name:
                    param.data[H:2*H].fill_(self.z_bias_input)
                elif 'bias_hh' in name:
                    param.data[H:2*H].fill_(self.z_bias_hidden)

        # Initialize output projection
        if not isinstance(self.to_out, nn.Identity):
            nn.init.constant_(self.to_out.weight, 0.0)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False, token_ids=None):
        """
        Forward pass with optional document boundary handling.

        Args:
            x: Input [B, T, D]
            prev_hidden: Initial hidden state [B, H] or None
            return_next_prev_hidden: Return final hidden state
            token_ids: Optional [B, T] token IDs for document boundary detection (0x1e = 30)
        """
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Project input
        x_proj = self.input_projection(x)  # [B, T, dim_inner]

        # Prepare initial hidden state
        if prev_hidden is None:
            h0 = torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)
        else:
            if prev_hidden.dim() == 3 and prev_hidden.size(1) == 1:
                prev_hidden = prev_hidden.squeeze(1)
            h0 = prev_hidden.unsqueeze(0) if prev_hidden.shape[0] == B else torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)

        # Process sequence
        if token_ids is None:
            # Fast path: no document boundaries, use cuDNN directly
            gru_out, hn = self.gru(x_proj, h0)  # Single fused cuDNN kernel!
        else:
            # Handle document boundaries by chunking
            # Find document boundaries (token_id == 30 for 0x1e)
            boundaries = (token_ids == 30)  # [B, T]

            # If no boundaries, use fast path
            if not boundaries.any():
                gru_out, hn = self.gru(x_proj, h0)
            else:
                # Process in chunks, resetting hidden state at boundaries
                # This is more complex but handles document boundaries correctly
                gru_out = []
                h_final = []

                for b in range(B):
                    # Find boundaries for this batch element
                    batch_boundaries = boundaries[b].nonzero(as_tuple=True)[0]

                    if len(batch_boundaries) == 0:
                        # No boundaries for this batch element
                        out_b, h_b = self.gru(x_proj[b:b+1], h0[:, b:b+1])
                        gru_out.append(out_b)
                        h_final.append(h_b)
                    else:
                        # Process in chunks between boundaries
                        out_chunks = []
                        h_b = h0[:, b:b+1].clone()  # Clone to avoid inplace issues
                        start = 0

                        for boundary_idx in batch_boundaries:
                            boundary_idx = boundary_idx.item()
                            if boundary_idx > start:
                                # Process chunk before boundary
                                chunk = x_proj[b:b+1, start:boundary_idx]
                                out_chunk, h_b = self.gru(chunk, h_b)
                                out_chunks.append(out_chunk)

                            # Reset hidden state at boundary
                            h_b = torch.zeros_like(h_b)

                            # Add boundary token (processed with reset hidden state)
                            chunk = x_proj[b:b+1, boundary_idx:boundary_idx+1]
                            out_chunk, h_b = self.gru(chunk, h_b)
                            out_chunks.append(out_chunk)

                            start = boundary_idx + 1

                        # Process remaining chunk
                        if start < T:
                            chunk = x_proj[b:b+1, start:T]
                            out_chunk, h_b = self.gru(chunk, h_b)
                            out_chunks.append(out_chunk)

                        # Concatenate chunks
                        out_b = torch.cat(out_chunks, dim=1)
                        gru_out.append(out_b)
                        h_final.append(h_b)

                # Stack batch dimension
                gru_out = torch.cat(gru_out, dim=0)
                hn = torch.cat(h_final, dim=1)

        # Apply output projection and residual
        out = self.to_out(gru_out) + x

        if return_next_prev_hidden:
            return out, hn.squeeze(0)
        return out


if __name__ == "__main__":
    print("Testing CuDNNFusedGRU...")

    device = 'cuda'
    B, T, D = 8, 64, 256

    model = CuDNNFusedGRU(D).to(device).to(torch.bfloat16)
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16) * 0.1

    # Test without document boundaries
    print("\nTest 1: Without document boundaries")
    out = model(x)
    print(f"✓ Output shape: {out.shape}")

    # Test with document boundaries
    print("\nTest 2: With document boundaries")
    token_ids = torch.randint(0, 100, (B, T), device=device)
    token_ids[:, 32] = 30  # Add document boundary in middle

    out_with_boundaries = model(x, token_ids=token_ids)
    print(f"✓ Output shape: {out_with_boundaries.shape}")

    # Test backward
    print("\nTest 3: Backward pass")
    loss = out_with_boundaries.sum()
    loss.backward()
    print(f"✓ Backward pass successful!")

    # Benchmark vs HybridFusedGRU
    print("\nBenchmark: CuDNNFusedGRU vs HybridFusedGRU")
    from mingru.hybrid_fused_gru import HybridFusedGRU
    import time

    hybrid_model = HybridFusedGRU(D).to(device).to(torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = model(x)
        _ = hybrid_model(x)
    torch.cuda.synchronize()

    # Benchmark cuDNN
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    cudnn_time = time.time() - start

    # Benchmark Hybrid
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = hybrid_model(x)
    torch.cuda.synchronize()
    hybrid_time = time.time() - start

    print(f"CuDNNFusedGRU: {cudnn_time:.3f}s")
    print(f"HybridFusedGRU: {hybrid_time:.3f}s")
    print(f"Speedup: {hybrid_time/cudnn_time:.2f}x")

    print("\n✓ All tests passed!")

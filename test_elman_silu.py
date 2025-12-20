"""
Test and benchmark the SiLU-gated Elman RNN (ElmanSilu).

ElmanSilu is designed for cuDNN-level performance with:
- 1 recurrent matmul per timestep (instead of 2 for expansion-based Elman)
- SiLU gating for selectivity
- Pre-computed input projections

Architecture:
  raw = R @ h + Wx @ x + b      -- [B, 2D] single recurrent matmul
  [h_candidate, gate_logit] = split(raw)
  h_candidate = tanh(h_candidate)
  gate = silu(gate_logit)
  h_new = h_candidate * gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from haste_pytorch import ElmanSilu, Elman, GRU


class ElmanSiluLM(nn.Module):
    """Language model using SiLU-gated Elman RNN."""

    def __init__(self, num_tokens: int, dim: int, depth: int):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([ElmanSilu(dim, dim) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, x):
        h = self.token_emb(x)  # [B, T, D]

        # Convert to time-first for RNN
        h = h.transpose(0, 1)  # [T, B, D]

        for layer, norm in zip(self.layers, self.norms):
            # Pre-norm residual
            h_norm = norm(h)
            h = h + layer(h_norm)

        # Convert back to batch-first
        h = h.transpose(0, 1)  # [B, T, D]

        h = self.norm_f(h)
        logits = self.to_logits(h)

        targets = x[:, 1:].contiguous()
        logits_for_loss = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), targets.view(-1))

        return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_basic():
    """Test basic forward/backward pass."""
    print("Testing basic forward/backward pass...")

    device = 'cuda'
    dtype = torch.float32

    B, T, D = 4, 32, 128

    layer = ElmanSilu(D, D).to(device, dtype)
    layer.train()

    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)

    # Forward pass
    out = layer(x)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check outputs
    assert out.shape == (T, B, D), f"Output shape mismatch: {out.shape}"
    assert x.grad is not None, "Input gradient is None"
    assert layer.Wx.grad is not None, "Wx gradient is None"
    assert layer.R.grad is not None, "R gradient is None"
    assert layer.bias.grad is not None, "bias gradient is None"

    print(f"  Output shape: {out.shape}")
    print(f"  Input grad norm: {x.grad.norm().item():.4f}")
    print(f"  Wx grad norm: {layer.Wx.grad.norm().item():.4f}")
    print(f"  R grad norm: {layer.R.grad.norm().item():.4f}")
    print(f"  bias grad norm: {layer.bias.grad.norm().item():.4f}")

    print("  Basic test PASSED!")
    return True


def test_gradient_vs_reference():
    """Test gradients against pure PyTorch reference implementation."""
    print("\nTesting gradients vs PyTorch reference...")

    device = 'cuda'
    dtype = torch.float64  # Use float64 for numerical stability

    B, T, D = 2, 4, 8

    layer = ElmanSilu(D, D).to(device, dtype)
    layer.train()

    # Extract weights
    Wx = layer.Wx.detach().clone()
    R = layer.R.detach().clone()
    bias = layer.bias.detach().clone()

    # Pure PyTorch reference
    def forward_ref(x, h0):
        h = h0
        outputs = []
        for t in range(x.shape[0]):
            raw = h @ R.t() + x[t] @ Wx.t() + bias
            h_cand = torch.tanh(raw[:, :D])
            gate = F.silu(raw[:, D:])
            h = h_cand * gate
            outputs.append(h)
        return torch.stack(outputs, dim=0)

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # ElmanSilu gradient
    x1 = x.clone().requires_grad_(True)
    out1 = layer(x1)
    loss1 = out1.sum()
    loss1.backward()
    grad_haste = x1.grad.clone()

    # PyTorch reference gradient
    x2 = x.clone().requires_grad_(True)
    out2 = forward_ref(x2, h0)
    loss2 = out2.sum()
    loss2.backward()
    grad_ref = x2.grad.clone()

    # Compare
    max_diff = (grad_haste - grad_ref).abs().max().item()
    rel_diff = max_diff / (grad_ref.abs().max().item() + 1e-10)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-4:
        print("  Gradient test PASSED!")
        return True
    else:
        print("  Gradient test FAILED!")
        return False


def benchmark_single_layer():
    """Benchmark a single ElmanSilu layer vs Elman and GRU."""
    print("\nBenchmarking single layer (T=512, B=64, D=2048)...")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.float16

    B, T, D = 64, 512, 2048

    # Create layers
    elman_silu = ElmanSilu(D, D).to(device, dtype)
    elman = Elman(D, D, expansion=2.0).to(device, dtype)
    gru = GRU(D, D).to(device, dtype)

    x = torch.randn(T, B, D, device=device, dtype=dtype)

    # Warmup
    for layer in [elman_silu, elman, gru]:
        layer.train()
        for _ in range(3):
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()
            layer.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    num_iters = 10
    results = {}

    for name, layer in [('ElmanSilu', elman_silu), ('Elman', elman), ('GRU', gru)]:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]  # GRU returns (output, h_n)
            loss = out.sum()
            loss.backward()
            layer.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        tokens = B * T * num_iters
        tok_per_sec = tokens / elapsed
        ms_per_iter = elapsed / num_iters * 1000

        results[name] = tok_per_sec
        print(f"  {name}: {tok_per_sec:,.0f} tok/s ({ms_per_iter:.1f} ms/iter)")

    return results


def benchmark_lm():
    """Benchmark full LM with ElmanSilu."""
    print("\nBenchmarking full LM (B=64, T=512, D=2048, depth=27)...")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.float16

    B, T = 64, 512
    num_tokens = 50281
    dim = 2048
    depth = 27

    print(f"Config: B={B}, T={T}, dim={dim}, depth={depth}")

    model = ElmanSiluLM(num_tokens, dim, depth).to(device, dtype)
    model.train()

    params = count_params(model)
    print(f"Model: {params:,} params ({params/1e9:.2f}B)")

    x = torch.randint(0, num_tokens, (B, T), device=device)

    # Warmup
    print("\nWarming up...")
    torch.cuda.synchronize()
    for i in range(3):
        loss = model(x)
        loss.backward()
        model.zero_grad()
        print(f"  Warmup iter {i+1}: loss={loss.item():.4f}")
    torch.cuda.synchronize()

    # Benchmark
    print("\nBenchmarking...")
    torch.cuda.synchronize()
    start = time.time()
    num_iters = 5
    for _ in range(num_iters):
        loss = model(x)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = B * T * num_iters
    tok_per_sec = tokens / elapsed
    ms_per_iter = elapsed / num_iters * 1000

    print(f"\nResults:")
    print(f"  Speed: {tok_per_sec:,.0f} tok/s ({ms_per_iter:.0f} ms/iter)")
    print(f"  Final loss: {loss.item():.4f}")

    print(f"\nComparison baseline:")
    print(f"  cuDNN GRU (fp16):    ~14,000 tok/s")
    print(f"  Haste Elman (B=64):  ~6,000 tok/s")
    print(f"  Haste ElmanSilu:     {tok_per_sec:,.0f} tok/s")

    return tok_per_sec


if __name__ == "__main__":
    print("Testing Haste ElmanSilu RNN")
    print("=" * 60)

    # Run tests
    test_basic()
    test_gradient_vs_reference()

    # Run benchmarks
    benchmark_single_layer()
    benchmark_lm()

    print("\nDone!")

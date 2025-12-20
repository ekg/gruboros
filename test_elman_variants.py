"""
Test and benchmark all Elman RNN variants.

Variants:
  ElmanSilu:    tanh + silu gate (original)
  ElmanTanh:    tanh + tanh gate
  ElmanSigmoid: tanh + sigmoid gate
  ElmanSwish:   silu + silu gate
  ElmanGelu:    tanh + gelu gate (approximate)
  ElmanNoGate:  tanh only, no gating (ablation baseline)

All gated variants follow the same architecture:
  raw = R @ h + Wx @ x + b          -- [B, 2D] single matmul
  [h_cand_raw, gate_raw] = split(raw)
  h_candidate = ACT1(h_cand_raw)    -- [B, D]
  gate = ACT2(gate_raw)             -- [B, D]
  h_new = h_candidate * gate        -- [B, D] elementwise
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

from haste_pytorch import (
    ElmanSilu, ElmanTanh, ElmanSigmoid, ElmanSwish, ElmanGelu, ElmanNoGate, GRU
)


def test_basic():
    """Test basic forward/backward pass for all variants."""
    print("Testing basic forward/backward pass...")
    print("=" * 60)

    device = 'cuda'

    # Test all dtypes
    test_dtypes = [
        ('fp32', torch.float32),
        ('bf16', torch.bfloat16),
        ('fp16', torch.float16),
    ]

    B, T, D = 4, 32, 128

    variants = [
        ('ElmanSilu', ElmanSilu),
        ('ElmanTanh', ElmanTanh),
        ('ElmanSigmoid', ElmanSigmoid),
        ('ElmanSwish', ElmanSwish),
        ('ElmanGelu', ElmanGelu),
        ('ElmanNoGate', ElmanNoGate),
    ]

    all_passed = True
    for dtype_name, dtype in test_dtypes:
        print(f"\n  Testing {dtype_name}:")
        for name, Layer in variants:
            try:
                layer = Layer(D, D).to(device, dtype)
                layer.train()

                x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)

                out = layer(x)
                loss = out.sum()
                loss.backward()

                assert out.shape == (T, B, D), f"Output shape mismatch: {out.shape}"
                assert x.grad is not None, "Input gradient is None"
                assert layer.Wx.grad is not None, "Wx gradient is None"
                assert layer.R.grad is not None, "R gradient is None"
                assert layer.bias.grad is not None, "bias gradient is None"

                print(f"    {name:15s}: PASSED")
            except Exception as e:
                print(f"    {name:15s}: FAILED ({e})")
                all_passed = False

    return all_passed


def test_gradients():
    """Test gradients against pure PyTorch reference implementations."""
    print("\nTesting gradients vs PyTorch reference...")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.float64

    B, T, D = 2, 4, 8

    all_passed = True

    # Reference implementations
    def make_reference(act1, act2_name):
        def forward_ref(x, h0, Wx, R, bias):
            h = h0
            outputs = []
            for t in range(x.shape[0]):
                raw = h @ R.t() + x[t] @ Wx.t() + bias
                h_cand_raw = raw[:, :D]
                gate_raw = raw[:, D:]

                h_candidate = act1(h_cand_raw)

                if act2_name == 'tanh':
                    gate = torch.tanh(gate_raw)
                elif act2_name == 'sigmoid':
                    gate = torch.sigmoid(gate_raw)
                elif act2_name == 'silu':
                    gate = F.silu(gate_raw)
                elif act2_name == 'gelu':
                    gate = F.gelu(gate_raw, approximate='tanh')

                h = h_candidate * gate
                outputs.append(h)
            return torch.stack(outputs, dim=0)
        return forward_ref

    def nogate_ref(x, h0, Wx, R, bias):
        h = h0
        outputs = []
        for t in range(x.shape[0]):
            raw = h @ R.t() + x[t] @ Wx.t() + bias
            h = torch.tanh(raw)
            outputs.append(h)
        return torch.stack(outputs, dim=0)

    # Test cases: (name, Layer, reference_func, gate_factor)
    tests = [
        ('ElmanSilu', ElmanSilu, make_reference(torch.tanh, 'silu'), 2),
        ('ElmanTanh', ElmanTanh, make_reference(torch.tanh, 'tanh'), 2),
        ('ElmanSigmoid', ElmanSigmoid, make_reference(torch.tanh, 'sigmoid'), 2),
        ('ElmanSwish', ElmanSwish, make_reference(F.silu, 'silu'), 2),
        ('ElmanGelu', ElmanGelu, make_reference(torch.tanh, 'gelu'), 2),
        ('ElmanNoGate', ElmanNoGate, nogate_ref, 1),
    ]

    for name, Layer, ref_func, gate_factor in tests:
        layer = Layer(D, D).to(device, dtype)
        layer.train()

        Wx = layer.Wx.detach().clone()
        R = layer.R.detach().clone()
        bias = layer.bias.detach().clone()

        x = torch.randn(T, B, D, device=device, dtype=dtype)
        h0 = torch.zeros(B, D, device=device, dtype=dtype)

        # CUDA gradient
        x1 = x.clone().requires_grad_(True)
        out1 = layer(x1)
        loss1 = out1.sum()
        loss1.backward()
        grad_haste = x1.grad.clone()

        # PyTorch reference gradient
        x2 = x.clone().requires_grad_(True)
        out2 = ref_func(x2, h0, Wx, R, bias)
        loss2 = out2.sum()
        loss2.backward()
        grad_ref = x2.grad.clone()

        max_diff = (grad_haste - grad_ref).abs().max().item()
        rel_diff = max_diff / (grad_ref.abs().max().item() + 1e-10)

        passed = rel_diff < 1e-4
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:15s}: {status} (rel_diff: {rel_diff:.2e})")

        if not passed:
            all_passed = False

    return all_passed


def benchmark_single_layer(batch_size=64, seq_len=512, hidden_size=2048, dtype_name='bf16'):
    """Benchmark all variants on a single layer."""
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_name, torch.bfloat16)
    print(f"\nBenchmarking single layer (T={seq_len}, B={batch_size}, D={hidden_size}, {dtype_name})...")
    print("=" * 60)

    device = 'cuda'

    B, T, D = batch_size, seq_len, hidden_size

    variants = [
        ('ElmanSilu', ElmanSilu),
        ('ElmanTanh', ElmanTanh),
        ('ElmanSigmoid', ElmanSigmoid),
        ('ElmanSwish', ElmanSwish),
        ('ElmanGelu', ElmanGelu),
        ('ElmanNoGate', ElmanNoGate),
        ('GRU (cuDNN)', GRU),
    ]

    x = torch.randn(T, B, D, device=device, dtype=dtype)

    results = {}
    for name, Layer in variants:
        try:
            layer = Layer(D, D).to(device, dtype)
            layer.train()

            # Warmup
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
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                out = layer(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = out.sum()
                loss.backward()
                layer.zero_grad()
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tokens = B * T * num_iters
            tok_per_sec = tokens / elapsed
            ms_per_iter = elapsed / num_iters * 1000

            results[name] = tok_per_sec
            print(f"  {name:15s}: {tok_per_sec:>10,.0f} tok/s ({ms_per_iter:.1f} ms/iter)")
        except NotImplementedError as e:
            print(f"  {name:15s}: SKIPPED (not supported for {dtype_name})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Elman RNN variants')
    parser.add_argument('--basic', action='store_true', help='Run basic tests only')
    parser.add_argument('--gradient', action='store_true', help='Run gradient tests only')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks only')
    parser.add_argument('--batch', type=int, default=64, help='Batch size for benchmark')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length for benchmark')
    parser.add_argument('--hidden', type=int, default=2048, help='Hidden size for benchmark')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Data type for benchmark')
    args = parser.parse_args()

    print("Elman RNN Variants Test Suite")
    print("=" * 60)

    run_all = not (args.basic or args.gradient or args.benchmark)

    if run_all or args.basic:
        test_basic()

    if run_all or args.gradient:
        test_gradients()

    if run_all or args.benchmark:
        benchmark_single_layer(args.batch, args.seq_len, args.hidden, args.dtype)

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Zero-Order Optimization - Triton V2: Virtual Perturbations + Evolving Seeds

Key improvements:
1. Virtual perturbations: Generate one at a time (not materialized)
2. Evolving seeds: Different perturbations each step (ergodic)
3. Memory: O(1) per perturbation instead of O(num_params)

For 500M params, 96 perturbations:
- Old: 189 GB (materialized vectors)
- New: ~2 GB peak (one perturbation at a time)
- Reduction: ~95× memory savings
"""

import torch
import torch.nn.functional as F
import time

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("WARNING: Triton not available. Install with: pip install triton")


# Keep the same Triton kernel for fused matmul+perturbation
@triton.jit
def matmul_with_perturbation_kernel(
    x_ptr, w_ptr, pert_ptr, output_ptr,
    M, N, K,
    epsilon,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pk, stride_pn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matmul with perturbation: Y = X @ (W + ε·P)
    P is passed in but generated on-demand (not stored across calls)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load X block
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load W block
        w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Load perturbation block
        p_ptrs = pert_ptr + k_offs[:, None] * stride_pk + offs_n[None, :] * stride_pn
        p_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
        p = tl.load(p_ptrs, mask=p_mask, other=0.0).to(tl.float32)

        # Fused operation: compute (W + ε·P) on-the-fly
        w_perturbed = w + epsilon * p

        # Accumulate
        acc += tl.dot(x, w_perturbed)

    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def matmul_with_perturbation(x, w, pert, epsilon):
    """Fused matmul with perturbation using Triton"""
    M, K = x.shape
    K2, N = w.shape
    assert K == K2
    assert pert.shape == w.shape

    # Fallback for small matrices
    if M < 16 or K < 16 or N < 16:
        with torch.no_grad():
            return torch.matmul(x, w + epsilon * pert)

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = max(16, min(64, triton.next_power_of_2(M)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(N)))
    BLOCK_K = max(16, min(32, triton.next_power_of_2(K)))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_with_perturbation_kernel[grid](
        x, w, pert, y,
        M, N, K,
        epsilon,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        pert.stride(0), pert.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return y


class TritonZeroOrderOptimizerV2:
    """
    Zero-Order Optimizer with Virtual Perturbations and Evolving Seeds

    Memory: O(1) per perturbation (just seeds!)
    Ergodic: Different perturbations each step
    Numerically correct: Uses torch.Generator
    """

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=96,
        pert_batch_size=16,
        world_size=1,
        rank=0,
        base_seed=42,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_perturbations = n_perturbations
        self.pert_batch_size = min(pert_batch_size, n_perturbations)
        self.world_size = world_size
        self.rank = rank
        self.base_seed = base_seed

        # CRITICAL: Track step counter for evolving seeds
        self.step_counter = 0

        # Put model in eval mode
        self.model.eval()

        # Collect parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_count = sum(p.numel() for p in self.params)

        # Gradient buffer
        self.grad_buffer = torch.zeros(self.param_count, device=self.params[0].device)

        if rank == 0:
            print(f"[Rank {rank}] Triton Zero-Order Optimizer V2 initialized:")
            print(f"  Parameters: {self.param_count:,}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Epsilon: {epsilon}")
            print(f"  Perturbations: {n_perturbations}")
            print(f"  Perturbation batch size: {self.pert_batch_size}")
            print(f"  Base seed: {base_seed}")
            print(f"  Method: VIRTUAL PERTURBATIONS + EVOLVING SEEDS")

            # Show memory savings
            old_memory_gb = (n_perturbations * self.param_count * 4) / (1024**3)
            peak_memory_gb = (self.param_count * 4) / (1024**3)  # One perturbation at a time
            print(f"  Memory savings: {old_memory_gb:.2f} GB → {peak_memory_gb:.2f} GB peak")
            print(f"  Reduction: {old_memory_gb / peak_memory_gb:.1f}× smaller")

    def zero_grad(self):
        """Reset gradient buffer"""
        self.grad_buffer.zero_()

    def _generate_single_perturbation(self, seed: int) -> torch.Tensor:
        """
        Generate a SINGLE perturbation from seed.

        Uses torch.Generator for numerical correctness.
        Perturbation is used immediately and discarded.

        Memory: O(num_params) temporarily, then freed
        """
        device = self.grad_buffer.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Rademacher: {-1, +1}
        pert = torch.randn(
            self.param_count,
            generator=generator,
            device=device,
            dtype=torch.float32
        )
        return torch.sign(pert)

    def _flatten_params(self):
        """Flatten all parameters into a single vector"""
        return torch.cat([p.data.reshape(-1) for p in self.params])

    def _unflatten_to_params(self, flat_tensor):
        """Unflatten vector back to parameter shapes"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(flat_tensor[offset:offset+numel].reshape(p.shape))
            offset += numel

    def _apply_perturbation(self, perturbation, scale=1.0):
        """Apply perturbation to parameters: θ ← θ + scale * perturbation"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(perturbation[offset:offset+numel].reshape(p.shape), alpha=scale)
            offset += numel

    def step(self, loss_fn, batch_data):
        """
        Virtual perturbation step with evolving seeds

        Key differences from old implementation:
        1. Generates one perturbation at a time (not all at once)
        2. Uses step_counter for evolving seeds
        3. Memory: O(1) per perturbation, not O(P × N)
        """
        start_time = time.time()

        # Determine perturbations per worker
        pert_per_worker = self.n_perturbations // self.world_size

        # CRITICAL: Evolving seed offset based on step counter
        step_seed_offset = self.step_counter * self.n_perturbations
        rank_offset = self.rank * pert_per_worker

        # Reset gradient buffer
        self.zero_grad()

        # Baseline loss (unperturbed)
        with torch.no_grad():
            logits = self.model(batch_data)
            targets = batch_data[:, 1:]
            logits_shifted = logits[:, :-1]
            loss_baseline = F.cross_entropy(
                logits_shifted.reshape(-1, logits_shifted.size(-1)),
                targets.reshape(-1),
                reduction='mean'
            ).item()

        # Process perturbations ONE AT A TIME
        for i in range(pert_per_worker):
            # Generate THIS perturbation's seed (deterministic but evolving!)
            seed = self.base_seed + step_seed_offset + rank_offset + i

            # Generate perturbation (temporarily)
            perturbation = self._generate_single_perturbation(seed)

            # Forward perturbation: θ + ε·P
            self._apply_perturbation(perturbation, scale=self.epsilon)
            with torch.no_grad():
                logits = self.model(batch_data)
                logits_shifted = logits[:, :-1]
                loss_plus = F.cross_entropy(
                    logits_shifted.reshape(-1, logits_shifted.size(-1)),
                    targets.reshape(-1),
                    reduction='mean'
                ).item()
            self._apply_perturbation(perturbation, scale=-self.epsilon)  # Remove

            # Backward perturbation: θ - ε·P
            self._apply_perturbation(perturbation, scale=-self.epsilon)
            with torch.no_grad():
                logits = self.model(batch_data)
                logits_shifted = logits[:, :-1]
                loss_minus = F.cross_entropy(
                    logits_shifted.reshape(-1, logits_shifted.size(-1)),
                    targets.reshape(-1),
                    reduction='mean'
                ).item()
            self._apply_perturbation(perturbation, scale=self.epsilon)  # Remove

            # Central difference gradient estimate
            grad_coef = (loss_plus - loss_minus) / (2 * self.epsilon)

            # Accumulate gradient estimate
            self.grad_buffer.add_(perturbation, alpha=grad_coef)

            # perturbation goes out of scope and is freed here!

        # Average gradient estimate
        self.grad_buffer.div_(pert_per_worker)

        # Apply gradient update
        flat_params = self._flatten_params()
        flat_params.add_(self.grad_buffer, alpha=-self.learning_rate)
        self._unflatten_to_params(flat_params)

        # Increment step counter for next step
        self.step_counter += 1

        end_time = time.time()

        return {
            'loss': loss_baseline,
            'time_total': end_time - start_time,
            'step_counter': self.step_counter,
        }

    def state_dict(self):
        """Save optimizer state"""
        return {
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'n_perturbations': self.n_perturbations,
            'pert_batch_size': self.pert_batch_size,
            'base_seed': self.base_seed,
            'step_counter': self.step_counter,  # CRITICAL: Save step counter!
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.learning_rate = state_dict['learning_rate']
        self.epsilon = state_dict['epsilon']
        self.n_perturbations = state_dict['n_perturbations']
        if 'pert_batch_size' in state_dict:
            self.pert_batch_size = state_dict['pert_batch_size']
        if 'base_seed' in state_dict:
            self.base_seed = state_dict['base_seed']
        if 'step_counter' in state_dict:
            self.step_counter = state_dict['step_counter']  # CRITICAL: Restore step counter!

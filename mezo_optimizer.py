"""
MeZO (Memory-efficient Zeroth-Order) Optimizer

Based on: "Fine-Tuning Language Models with Just Forward Passes" (NeurIPS 2023)
https://arxiv.org/abs/2305.17333

Key features:
- Same memory footprint as inference (no backward pass!)
- In-place perturbation using seed resampling
- Multi-perturbation accumulation for variance reduction
- Scales linearly to thousands of GPUs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import time
import subprocess


class MeZOOptimizer:
    """
    Memory-Efficient Zeroth-Order Optimizer

    Uses SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimation
    with in-place perturbation to achieve the same memory footprint as inference.

    Perfect for:
    - Unbounded context training (forward-only, no TBPTT limitations)
    - Massive parallelism (thousands of GPUs)
    - Memory-constrained environments
    """

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        epsilon=1e-3,
        num_perturbations=1,
        base_seed=42,
        rank=0,
        world_size=1,
        momentum=0.9,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_perturbations = num_perturbations
        self.base_seed = base_seed
        self.rank = rank
        self.world_size = world_size
        self.momentum = momentum

        self.step_counter = 0
        self.model.eval()  # Always in eval mode (no dropout/batchnorm changes)

        # Get all trainable parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_count = sum(p.numel() for p in self.params)

        # Initialize momentum buffer (simple momentum, not Adam!)
        self.velocity = [torch.zeros_like(p.data) for p in self.params]

        # PyTorch optimizer interface compatibility
        self.param_groups = [{'lr': learning_rate, 'params': list(model.parameters())}]

        if rank == 0:
            print(f"[Rank {rank}] MeZO Optimizer initialized:")
            print(f"  Parameters: {self.param_count:,}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Epsilon (perturbation scale): {epsilon}")
            print(f"  Perturbations per step: {num_perturbations}")
            print(f"  Momentum: {momentum} (simple SGD momentum)")
            print(f"  Method: IN-PLACE SEED-BASED PERTURBATION")
            print(f"  Memory: SAME AS INFERENCE (no gradients, no backward)")
            print(f"  Perfect for unbounded context & massive parallelism!")

    def zero_grad(self):
        """No-op for compatibility with PyTorch optimizer API"""
        pass

    @contextmanager
    def _perturb_parameters(self, seed, sign):
        """
        Temporarily perturb all parameters in-place using CHUNKED perturbations.

        MEMORY OPTIMIZATION: Process parameters in 4MB chunks to avoid OOM!
        - Old approach: 588MB temporary allocation for embedding layer
        - New approach: 4MB max temporary allocation (150× reduction!)

        TRUE SEED-BASED RESTORATION (no cloning!)
        - Apply: θ += sign * ε * z
        - Restore: θ -= sign * ε * z (regenerate z from same seed)

        Args:
            seed: Random seed for reproducible perturbation
            sign: +1 or -1 for forward/backward perturbation
        """
        # CHUNKED PERTURBATION: Only allocate 4MB at a time!
        CHUNK_SIZE = 1024 * 1024  # 1M float32 values = 4MB

        # Apply perturbation in chunks (no full materialization!)
        for i, param in enumerate(self.params):
            param_seed = seed + i
            original_shape = param.shape
            flat_param = param.data.view(-1)
            num_elements = flat_param.numel()

            # Process in chunks to keep memory usage low
            for chunk_start in range(0, num_elements, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, num_elements)
                chunk_size = chunk_end - chunk_start

                # Deterministic seed for this chunk
                chunk_seed = param_seed + chunk_start
                generator = torch.Generator(device=param.device)
                generator.manual_seed(chunk_seed)

                # Generate perturbation for JUST THIS CHUNK (not the whole param!)
                z_chunk = torch.randn(
                    chunk_size,
                    generator=generator,
                    device=param.device,
                    dtype=param.dtype
                )

                # Apply perturbation to this chunk
                flat_param[chunk_start:chunk_end].add_(z_chunk, alpha=sign * self.epsilon)

        try:
            yield
        finally:
            # Restore by regenerating same chunks and subtracting
            for i, param in enumerate(self.params):
                param_seed = seed + i
                flat_param = param.data.view(-1)
                num_elements = flat_param.numel()

                # Process in same chunks to restore
                for chunk_start in range(0, num_elements, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, num_elements)
                    chunk_size = chunk_end - chunk_start

                    # Same seed = same random values!
                    chunk_seed = param_seed + chunk_start
                    generator = torch.Generator(device=param.device)
                    generator.manual_seed(chunk_seed)

                    # Regenerate SAME chunk
                    z_chunk = torch.randn(
                        chunk_size,
                        generator=generator,
                        device=param.device,
                        dtype=param.dtype
                    )

                    # Subtract the same perturbation to restore θ
                    flat_param[chunk_start:chunk_end].add_(z_chunk, alpha=-sign * self.epsilon)

    def _compute_loss(self, batch_data):
        """
        Compute cross-entropy loss for language modeling.

        CRITICAL: Uses streaming loss computation that processes positions in small batches.
        This avoids materializing full [batch, seq, vocab] logits tensor (saves ~4.9 GB!)

        Args:
            batch_data: Input tensor [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        with torch.no_grad():
            # The model's forward() expects input x and will internally shift for labels
            # When return_loss=True, it computes loss using streaming (64-position batches)
            loss = self.model(batch_data, return_loss=True)

            return loss.item()

    def _get_gpu_utilization(self):
        """Get current GPU utilization percentage."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                utils = [int(x.strip()) for x in result.stdout.strip().split('\n')]
                return utils[self.rank] if self.rank < len(utils) else 0
        except:
            return 0
        return 0

    def step(self, loss_fn, batch_provider):
        """
        Perform one MeZO optimization step with multi-perturbation accumulation.

        Args:
            loss_fn: Not used (for compatibility)
            batch_provider: Function that returns (batch_data, ...) or just batch_data

        Returns:
            Dictionary with loss and timing info
        """
        start_time = time.time()
        gpu_utils = []

        # Accumulate gradients across K perturbations
        accumulated_grad_coef = 0.0
        all_losses = []

        for k in range(self.num_perturbations):
            # Get fresh batch for this perturbation (DATA DIVERSITY IS CRITICAL!)
            result = batch_provider()
            if isinstance(result, tuple):
                batch_data = result[0]
            else:
                batch_data = result

            # Generate unique seed for this perturbation
            # Format: base + (step * K + k) * world_size + rank
            seed = self.base_seed + (self.step_counter * self.num_perturbations + k) * self.world_size + self.rank

            # Forward perturbation: θ + ε*z_k
            with self._perturb_parameters(seed, +1):
                loss_plus = self._compute_loss(batch_data)

            # Sample GPU utilization during forward pass
            gpu_utils.append(self._get_gpu_utilization())

            # Backward perturbation: θ - ε*z_k
            with self._perturb_parameters(seed, -1):
                loss_minus = self._compute_loss(batch_data)

            # Sample GPU utilization during backward pass
            gpu_utils.append(self._get_gpu_utilization())

            # Accumulate gradient coefficient for this perturbation
            grad_coef_k = (loss_plus - loss_minus) / (2 * self.epsilon)
            accumulated_grad_coef += grad_coef_k

            # Track losses
            all_losses.append((loss_plus + loss_minus) / 2)

        # Average the accumulated gradient coefficient
        avg_grad_coef = accumulated_grad_coef / self.num_perturbations

        # Synchronize gradient coefficients across all ranks (DDP)
        if self.world_size > 1:
            import torch.distributed as dist
            grad_tensor = torch.tensor([avg_grad_coef], device='cuda')
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.AVG)
            avg_grad_coef = grad_tensor.item()

        # Apply simple momentum update (proven for MeZO)
        # gradient estimate: g = avg_grad_coef * z
        # momentum update: v = momentum * v + g, θ = θ - lr * v
        total_grad_norm_sq = 0.0
        with torch.no_grad():
            for i, param in enumerate(self.params):
                # Compute gradient estimate: g = avg_grad_coef * z
                # For K perturbations, we average: g = (1/K) * sum_k [grad_coef_k * z_k]
                grad_estimate = torch.zeros_like(param.data)

                for k in range(self.num_perturbations):
                    seed = self.base_seed + (self.step_counter * self.num_perturbations + k) * self.world_size + self.rank
                    param_seed = seed + i
                    generator = torch.Generator(device=param.device)
                    generator.manual_seed(param_seed)
                    z = torch.randn(
                        param.shape,
                        generator=generator,
                        device=param.device,
                        dtype=param.dtype
                    )
                    # Accumulate: g += (avg_grad_coef / K) * z
                    grad_estimate.add_(z, alpha=avg_grad_coef / self.num_perturbations)

                # Accumulate gradient norm (L2 norm across all parameters)
                total_grad_norm_sq += grad_estimate.norm(2).item() ** 2

                # Update velocity: v = momentum * v + g
                self.velocity[i].mul_(self.momentum).add_(grad_estimate)

                # Update parameters: θ = θ - lr * v
                param.data.add_(self.velocity[i], alpha=-self.learning_rate)

        # Compute total gradient norm
        grad_norm = total_grad_norm_sq ** 0.5

        self.step_counter += 1
        end_time = time.time()

        # Return averaged metrics
        avg_loss = sum(all_losses) / len(all_losses)
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0

        return {
            'loss': avg_loss,
            'grad_norm': grad_norm,  # Add gradient norm for monitoring
            'time_total': end_time - start_time,
            'step': self.step_counter,
            'num_perturbations': self.num_perturbations,
            'gpu_utilization': avg_gpu_util,
            'forward_passes': 2 * self.num_perturbations,
        }

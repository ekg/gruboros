"""
Zero-Order Optimization for Memory-Efficient Training
Implements CD-RGE (Central-Difference Random Gradient Estimation)

Based on: "Scaling Recurrent Neural Networks to a Billion Parameters
with Zero-Order Optimization" (arXiv:2505.17852)
"""

import torch
import torch.distributed as dist
import random
import time


class CD_RGE_Optimizer:
    """
    Central-Difference Random Gradient Estimation Optimizer

    Key features:
    - Memory efficient: No activation storage needed
    - Uses Rademacher probes for gradient estimation
    - Antithetic sampling for variance reduction
    - Distributed training support

    Args:
        model: PyTorch model to optimize
        learning_rate: Step size (recommended: equal to epsilon)
        epsilon: Perturbation size (recommended: equal to learning_rate)
        n_perturbations: Number of probe vectors (default: 96)
        world_size: Number of distributed workers
        rank: Current worker rank
    """

    def __init__(self, model, learning_rate=1e-4, epsilon=1e-4,
                 n_perturbations=96, world_size=1, rank=0, chunk_size=512):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_perturbations = n_perturbations
        self.world_size = world_size
        self.rank = rank
        self.chunk_size = chunk_size  # For memory-efficient sequential forward passes

        # Put model in eval mode (no batchnorm/dropout stochasticity)
        self.model.eval()

        # Count parameters
        self.param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Pre-allocate gradient buffer
        self.grad_buffer = torch.zeros(self.param_count, device=next(model.parameters()).device)

        # PyTorch optimizer interface compatibility
        self.param_groups = [{'lr': learning_rate, 'params': list(model.parameters())}]

        print(f"[Rank {rank}] CD-RGE Optimizer initialized:")
        print(f"  Parameters: {self.param_count:,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Perturbations: {n_perturbations}")
        print(f"  Forward passes per step: {2 * n_perturbations}")
        print(f"  Memory-efficient chunk size: {chunk_size} tokens")

    def zero_grad(self):
        """Reset gradient buffer (compatibility with PyTorch optimizer API)"""
        self.grad_buffer.zero_()

    def apply_probe(self, seed, scale=1.0):
        """
        Apply Rademacher probe to model parameters in-place.

        Args:
            seed: Random seed for reproducibility
            scale: Scaling factor (epsilon or -epsilon)
        """
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # Rademacher: {-1, +1} distribution
                    noise = torch.randint(0, 2, param.shape, device=param.device, dtype=param.dtype) * 2 - 1
                    param.add_(scale * noise)

    def restore_probe(self, seed, scale=1.0):
        """Remove probe from model parameters (reverse of apply_probe)"""
        self.apply_probe(seed, -scale)

    def reconstruct_probe_vector(self, seed):
        """
        Reconstruct full probe vector from seed.
        Returns flat probe vector matching param_count.
        """
        torch.manual_seed(seed)
        probe_parts = []

        for param in self.model.parameters():
            if param.requires_grad:
                noise = torch.randint(0, 2, param.shape, device=param.device, dtype=param.dtype) * 2 - 1
                probe_parts.append(noise.flatten())

        return torch.cat(probe_parts)

    def step(self, loss_fn, *args, **kwargs):
        """
        Perform one optimization step using CD-RGE.

        Args:
            loss_fn: Callable that computes loss given model output
            *args, **kwargs: Arguments passed to loss_fn

        Returns:
            dict with 'loss' (mean), 'loss_std', 'time_forward', 'time_backward'
        """
        start_time = time.time()

        # Determine perturbations per worker
        pert_per_worker = self.n_perturbations // self.world_size
        start_idx = self.rank * pert_per_worker

        # Collect losses from forward passes
        losses_plus = []
        losses_minus = []
        seeds = []

        forward_start = time.time()

        # Compute losses for each perturbation (antithetic pairs)
        for i in range(pert_per_worker):
            seed = start_idx + i
            seeds.append(seed)

            # Forward pass: θ + ε·p_i
            self.apply_probe(seed, self.epsilon)
            with torch.no_grad():
                loss_plus = loss_fn(*args, **kwargs)
            losses_plus.append(loss_plus.item())
            self.restore_probe(seed, self.epsilon)

            # Forward pass: θ - ε·p_i (antithetic)
            self.apply_probe(seed, -self.epsilon)
            with torch.no_grad():
                loss_minus = loss_fn(*args, **kwargs)
            losses_minus.append(loss_minus.item())
            self.restore_probe(seed, -self.epsilon)

        forward_time = time.time() - forward_start

        # Distributed: gather all losses at rank 0
        backward_start = time.time()

        if self.world_size > 1:
            # Gather losses from all workers
            if self.rank == 0:
                all_losses_plus = [None] * self.world_size
                all_losses_minus = [None] * self.world_size
            else:
                all_losses_plus = None
                all_losses_minus = None

            dist.gather_object(losses_plus, all_losses_plus, dst=0)
            dist.gather_object(losses_minus, all_losses_minus, dst=0)

            # Rank 0 computes gradient and broadcasts update
            if self.rank == 0:
                # Flatten gathered losses
                losses_plus = [item for sublist in all_losses_plus for item in sublist]
                losses_minus = [item for sublist in all_losses_minus for item in sublist]

                # Compute gradient estimate
                self._compute_gradient_estimate(losses_plus, losses_minus)

                # Apply update
                self._apply_gradient_update()

                # Broadcast updated parameters
                for param in self.model.parameters():
                    if param.requires_grad:
                        dist.broadcast(param.data, src=0)

                mean_loss = (sum(losses_plus) + sum(losses_minus)) / (2 * len(losses_plus))
                loss_std = torch.tensor([losses_plus + losses_minus]).std().item()
            else:
                # Workers receive updated parameters
                for param in self.model.parameters():
                    if param.requires_grad:
                        dist.broadcast(param.data, src=0)

                mean_loss = sum(losses_plus + losses_minus) / (2 * len(losses_plus))
                loss_std = 0.0
        else:
            # Single GPU: compute and apply gradient directly
            self._compute_gradient_estimate(losses_plus, losses_minus)
            self._apply_gradient_update()

            mean_loss = (sum(losses_plus) + sum(losses_minus)) / (2 * len(losses_plus))
            loss_std = torch.tensor([losses_plus + losses_minus]).std().item()

        backward_time = time.time() - backward_start
        total_time = time.time() - start_time

        return {
            'loss': mean_loss,
            'loss_std': loss_std,
            'time_forward': forward_time,
            'time_backward': backward_time,
            'time_total': total_time,
            'n_forward_passes': 2 * pert_per_worker
        }

    def _compute_gradient_estimate(self, losses_plus, losses_minus):
        """
        Compute gradient estimate from probe losses.
        Stores result in self.grad_buffer.
        """
        self.grad_buffer.zero_()

        for i in range(len(losses_plus)):
            # Reconstruct probe
            probe = self.reconstruct_probe_vector(i)

            # Central difference: (f(θ+ε·p) - f(θ-ε·p)) / (2ε)
            loss_diff = (losses_plus[i] - losses_minus[i]) / (2 * self.epsilon)

            # Gradient contribution: loss_diff * probe
            self.grad_buffer.add_(probe, alpha=loss_diff)

        # Average over all perturbations
        self.grad_buffer.div_(len(losses_plus))

    def _apply_gradient_update(self):
        """Apply gradient update: θ ← θ - η·∇f(θ)"""
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    param_numel = param.numel()
                    param_grad = self.grad_buffer[idx:idx+param_numel].view_as(param)
                    param.sub_(param_grad, alpha=self.learning_rate)
                    idx += param_numel

    def state_dict(self):
        """Return optimizer state (for checkpointing)"""
        return {
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'n_perturbations': self.n_perturbations
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state (from checkpoint)"""
        self.learning_rate = state_dict['learning_rate']
        self.epsilon = state_dict['epsilon']
        self.n_perturbations = state_dict['n_perturbations']


class ZeroOrderLossWrapper:
    """
    Wrapper for loss computation in zero-order training.

    Handles forward pass and loss computation without gradient tracking.
    """

    def __init__(self, model, criterion, chunk_size):
        self.model = model
        self.criterion = criterion
        self.chunk_size = chunk_size

    def __call__(self, tokens, is_doc_end=None, hidden_states=None):
        """
        Compute loss for a batch of tokens.

        Args:
            tokens: Input token IDs [batch, seq_len]
            is_doc_end: Document boundary markers [batch]
            hidden_states: Initial hidden states (optional)

        Returns:
            Scalar loss tensor
        """
        with torch.no_grad():
            # Forward pass
            if is_doc_end is not None:
                # Reset hidden states at document boundaries
                if hidden_states is not None:
                    batch_size = tokens.shape[0]
                    for b in range(batch_size):
                        if is_doc_end[b]:
                            for layer_hidden in hidden_states:
                                layer_hidden[b].zero_()

            # Get model outputs
            outputs = self.model(tokens[:, :-1])
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Compute cross-entropy loss
            targets = tokens[:, 1:]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            return loss

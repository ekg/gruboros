"""
Zero-Order Optimization - Option 1: vmap with functional_call

Uses torch.func.vmap to vectorize forward passes across perturbations.
True parallelization - all perturbations evaluated simultaneously.

Expected: 10× speedup over sequential (0.069s → 0.007s per step)
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap
import time


class VmapZeroOrderOptimizer:
    """
    Zero-Order Optimizer using vmap for true parallelization.

    All perturbations are evaluated in a single batched operation using vmap.
    No weight materialization - parameters created on-the-fly in vmap.

    Args:
        model: PyTorch model to optimize
        learning_rate: Step size
        epsilon: Perturbation size
        n_perturbations: Number of probe vectors (default: 96)
        pert_batch_size: Number of perturbations to process in parallel (default: 16)
    """

    def __init__(self, model, learning_rate=1e-4, epsilon=1e-4,
                 n_perturbations=96, pert_batch_size=16, world_size=1, rank=0):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_perturbations = n_perturbations
        self.pert_batch_size = pert_batch_size
        self.world_size = world_size
        self.rank = rank

        # Put model in eval mode
        self.model.eval()

        # Get base parameters as dict
        self.base_params = {name: param for name, param in model.named_parameters()
                           if param.requires_grad}

        # Count parameters
        self.param_count = sum(p.numel() for p in self.base_params.values())

        # Pre-allocate gradient buffer
        device = next(model.parameters()).device
        self.grad_buffer = torch.zeros(self.param_count, device=device)

        # PyTorch optimizer interface compatibility
        self.param_groups = [{'lr': learning_rate, 'params': list(model.parameters())}]

        print(f"[Rank {rank}] Vmap Zero-Order Optimizer initialized:")
        print(f"  Parameters: {self.param_count:,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Perturbations: {n_perturbations}")
        print(f"  Perturbation batch size: {pert_batch_size}")
        print(f"  Method: torch.func.vmap (TRUE PARALLELIZATION)")

    def zero_grad(self):
        """Reset gradient buffer"""
        self.grad_buffer.zero_()

    def generate_perturbations(self, n_pert, seed_offset=0):
        """
        Generate Rademacher perturbation vectors.

        Args:
            n_pert: Number of perturbations
            seed_offset: Offset for random seed

        Returns:
            Tensor of shape [n_pert, param_count] with {-1, +1} entries
        """
        device = self.grad_buffer.device
        perturbations = []

        for i in range(n_pert):
            torch.manual_seed(seed_offset + i)
            pert = torch.randint(0, 2, (self.param_count,), device=device, dtype=torch.float32)
            pert = pert * 2 - 1  # {0,1} -> {-1,+1}
            perturbations.append(pert)

        return torch.stack(perturbations)  # [n_pert, param_count]

    def create_perturbed_params_from_flat(self, base_params_dict, pert_flat, epsilon_scale):
        """
        Create perturbed parameter dict from flat perturbation vector.

        Args:
            base_params_dict: Dict of base parameters
            pert_flat: [param_count] flat perturbation vector
            epsilon_scale: +1 or -1

        Returns:
            Dict of perturbed parameters
        """
        param_dict = {}
        offset = 0

        for name, param in base_params_dict.items():
            numel = param.numel()
            pert_piece = pert_flat[offset:offset+numel].reshape(param.shape)
            param_dict[name] = param + epsilon_scale * self.epsilon * pert_piece
            offset += numel

        return param_dict

    def vmap_forward_batch(self, perturbations, epsilon_scale, batch_data):
        """
        Run forward passes in PARALLEL using vmap.

        This is the key function - uses vmap to vectorize over perturbation dimension.

        Args:
            perturbations: [n_pert, param_count] tensor
            epsilon_scale: +1 or -1
            batch_data: Input tensor [batch, seq_len]

        Returns:
            all_logits: [n_pert, batch, seq_len, vocab_size]
        """
        # Define function that runs forward pass for single perturbation
        def forward_single_pert(pert_flat):
            # Create perturbed params on-the-fly
            param_dict = self.create_perturbed_params_from_flat(
                self.base_params, pert_flat, epsilon_scale
            )

            # Run forward pass with perturbed params
            with torch.no_grad():
                outputs = functional_call(self.model, param_dict, (batch_data[:, :-1],))
                if isinstance(outputs, dict):
                    return outputs['logits']
                return outputs

        # Use vmap to vectorize over perturbation dimension
        # This runs ALL perturbations in parallel!
        try:
            all_logits = vmap(forward_single_pert)(perturbations)
        except Exception as e:
            print(f"Warning: vmap failed ({e}), falling back to loop")
            # Fallback: manual loop
            all_logits = []
            for i in range(perturbations.shape[0]):
                logits = forward_single_pert(perturbations[i])
                all_logits.append(logits)
            all_logits = torch.stack(all_logits)

        return all_logits  # [n_pert, batch, seq_len, vocab]

    def compute_losses_parallel(self, all_logits, targets):
        """
        Compute loss for each perturbation in parallel.

        Args:
            all_logits: [n_pert, batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            losses: [n_pert] - loss for each perturbation
        """
        n_pert = all_logits.shape[0]
        batch_size, seq_len = targets.shape
        vocab_size = all_logits.shape[-1]

        # Expand targets to match perturbation dimension
        targets_exp = targets.unsqueeze(0).expand(n_pert, -1, -1)

        # Flatten for cross-entropy
        logits_flat = all_logits.reshape(-1, vocab_size)
        targets_flat = targets_exp.reshape(-1)

        # Compute cross-entropy (no reduction yet)
        losses_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Reshape and average over batch and sequence dimensions
        losses = losses_flat.reshape(n_pert, -1).mean(dim=1)

        return losses

    def step(self, loss_fn, batch_data=None, *args, **kwargs):
        """
        Perform one optimization step using vmap-based parallel evaluation.

        Args:
            loss_fn: Callable that computes loss (not used in vmap version)
            batch_data: Input tensor [batch, seq_len]

        Returns:
            dict with 'loss', 'loss_std', 'time_forward', 'time_backward'
        """
        start_time = time.time()

        # Determine perturbations per worker
        pert_per_worker = self.n_perturbations // self.world_size
        start_idx = self.rank * pert_per_worker

        # Generate all perturbation vectors for this worker
        perturbations = self.generate_perturbations(pert_per_worker, seed_offset=start_idx)

        # Collect losses
        all_losses_plus = []
        all_losses_minus = []

        forward_start = time.time()

        # Process perturbations in mini-batches for memory efficiency
        n_batches = (pert_per_worker + self.pert_batch_size - 1) // self.pert_batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.pert_batch_size
            end = min(start + self.pert_batch_size, pert_per_worker)
            batch_perts = perturbations[start:end]

            # PARALLEL forward passes for +ε perturbations (using vmap!)
            logits_plus = self.vmap_forward_batch(batch_perts, epsilon_scale=+1.0, batch_data=batch_data)
            targets = batch_data[:, 1:]  # Shift for next-token prediction
            losses_plus = self.compute_losses_parallel(logits_plus, targets)
            all_losses_plus.extend(losses_plus.cpu().tolist())

            # PARALLEL forward passes for -ε perturbations (using vmap!)
            logits_minus = self.vmap_forward_batch(batch_perts, epsilon_scale=-1.0, batch_data=batch_data)
            losses_minus = self.compute_losses_parallel(logits_minus, targets)
            all_losses_minus.extend(losses_minus.cpu().tolist())

            # Clear memory
            del logits_plus, logits_minus
            torch.cuda.empty_cache()

        forward_time = time.time() - forward_start

        # Compute gradient estimate
        backward_start = time.time()
        self._compute_gradient_estimate(perturbations, all_losses_plus, all_losses_minus)

        # Apply gradient update
        self._apply_gradient_update()

        backward_time = time.time() - backward_start
        total_time = time.time() - start_time

        mean_loss = (sum(all_losses_plus) + sum(all_losses_minus)) / (2 * len(all_losses_plus))
        loss_std = torch.tensor(all_losses_plus + all_losses_minus).std().item()

        return {
            'loss': mean_loss,
            'loss_std': loss_std,
            'time_forward': forward_time,
            'time_backward': backward_time,
            'time_total': total_time,
            'n_forward_passes': 2 * pert_per_worker
        }

    def _compute_gradient_estimate(self, perturbations, losses_plus, losses_minus):
        """
        Compute gradient estimate from probe losses.

        Args:
            perturbations: [n_pert, param_count] tensor
            losses_plus: List of losses at θ + ε·p
            losses_minus: List of losses at θ - ε·p
        """
        self.grad_buffer.zero_()

        n_pert = len(losses_plus)
        for i in range(n_pert):
            # Central difference: (f(θ+ε·p) - f(θ-ε·p)) / (2ε)
            loss_diff = (losses_plus[i] - losses_minus[i]) / (2 * self.epsilon)

            # Gradient contribution: loss_diff * probe
            self.grad_buffer.add_(perturbations[i], alpha=loss_diff)

        # Average over all perturbations
        self.grad_buffer.div_(n_pert)

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
            'n_perturbations': self.n_perturbations,
            'pert_batch_size': self.pert_batch_size
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state (from checkpoint)"""
        self.learning_rate = state_dict['learning_rate']
        self.epsilon = state_dict['epsilon']
        self.n_perturbations = state_dict['n_perturbations']
        if 'pert_batch_size' in state_dict:
            self.pert_batch_size = state_dict['pert_batch_size']

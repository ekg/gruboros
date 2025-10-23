"""
Zero-Order Optimization with Layer-Wise Materialized Perturbations

Key idea: Materialize perturbations PER LAYER, not for entire model.
Memory: O(layer_size * n_perts) instead of O(model_size * n_perts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import time


@contextmanager
def layerwise_perturbed_forward(layer, seeds_list, epsilon):
    """
    Materialize perturbations for a single layer and apply batched computation.
    
    Args:
        layer: nn.Linear layer
        seeds_list: List of seeds [seed1, seed2, ..., seedN]
        epsilon: Perturbation magnitude
    """
    original_forward = layer.forward
    n_perts = len(seeds_list)
    device = layer.weight.device
    
    # Pre-generate perturbations for this layer
    # Shape: [n_perts, out_features, in_features] to match weight.t()
    weight_shape = layer.weight.shape  # [out_features, in_features]
    perturbations = []
    
    for seed in seeds_list:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        pert = torch.randn(weight_shape, generator=generator, device=device).sign()
        perturbations.append(pert)
    
    perturbations = torch.stack(perturbations)  # [n_perts, out, in]
    
    def batched_forward(x):
        """
        x shape: [n_perts * batch_size, seq_len, in_features] or [n_perts * batch_size, in_features]

        Each chunk of batch_size rows uses a different perturbation.
        Uses TRUE batched matmul - processes all perturbations in parallel!
        """
        original_shape = x.shape
        batch_size = original_shape[0] // n_perts

        # Stack perturbed weights: [n_perts, out_features, in_features]
        weights_perturbed = layer.weight.unsqueeze(0) + epsilon * perturbations

        # Reshape: [n_perts, batch_size, ...remaining dims..., in_features]
        if x.dim() == 3:
            seq_len = x.size(1)
            in_features = x.size(2)
            out_features = layer.weight.size(0)

            # x_batched: [n_perts, batch_size, seq_len, in_features]
            x_batched = x.reshape(n_perts, batch_size, seq_len, in_features)

            # Batched matmul using einsum:
            # [n_perts, batch_size, seq_len, in_features] @ [n_perts, out_features, in_features]
            # -> [n_perts, batch_size, seq_len, out_features]
            # Note: weight is [out_features, in_features], so we do einsum with 'oi' order
            output = torch.einsum('nbsi,noi->nbso', x_batched, weights_perturbed)

            # Add bias if present
            if layer.bias is not None:
                output = output + layer.bias.view(1, 1, 1, -1)

            # Reshape to original batch structure: [n_perts * batch_size, seq_len, out_features]
            output = output.reshape(n_perts * batch_size, seq_len, out_features)

        else:  # 2D input
            in_features = x.size(1)
            out_features = layer.weight.size(0)

            # x_batched: [n_perts, batch_size, in_features]
            x_batched = x.reshape(n_perts, batch_size, in_features)

            # Batched matmul: [n_perts, batch_size, in_features] @ [n_perts, out_features, in_features]
            # -> [n_perts, batch_size, out_features]
            output = torch.einsum('nbi,noi->nbo', x_batched, weights_perturbed)

            # Add bias if present
            if layer.bias is not None:
                output = output + layer.bias.view(1, 1, -1)

            # Reshape: [n_perts * batch_size, out_features]
            output = output.reshape(n_perts * batch_size, out_features)

        return output
    
    layer.forward = batched_forward
    
    try:
        yield
    finally:
        layer.forward = original_forward
        del perturbations
        torch.cuda.empty_cache()


class LayerwiseZeroOrderOptimizer:
    """
    Zero-Order Optimizer with layer-wise materialized perturbations.

    Materializes perturbations one layer at a time during forward pass.
    Processes perturbations in chunks to avoid OOM on large models.
    """

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        epsilon=1e-4,
        n_perturbations=8,
        perturbation_chunk_size=8,
        base_seed=42,
        rank=0,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_perturbations = n_perturbations
        self.perturbation_chunk_size = min(perturbation_chunk_size, n_perturbations)
        self.base_seed = base_seed
        self.rank = rank

        self.step_counter = 0
        self.model.eval()

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_count = sum(p.numel() for p in self.params)

        # PyTorch optimizer interface compatibility
        self.param_groups = [{'lr': learning_rate, 'params': list(model.parameters())}]

        if rank == 0:
            print(f"[Rank {rank}] Layer-wise Zero-Order Optimizer initialized:")
            print(f"  Parameters: {self.param_count:,}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Epsilon: {epsilon}")
            print(f"  Total perturbations: {n_perturbations}")
            print(f"  Perturbation chunk size: {self.perturbation_chunk_size}")
            print(f"  Method: CHUNKED LAYER-WISE MATERIALIZATION")

            # Calculate max layer memory PER CHUNK
            max_layer_size = max(p.numel() for p in self.params)
            mem_per_chunk_gb = (self.perturbation_chunk_size * max_layer_size * 4) / (1024**3)
            num_chunks = (n_perturbations + self.perturbation_chunk_size - 1) // self.perturbation_chunk_size
            print(f"  Max memory per chunk: {mem_per_chunk_gb:.2f} GB")
            print(f"  Number of chunks: {num_chunks}")
            print(f"  Memory freed after each chunk")

    def zero_grad(self):
        """No-op for compatibility with PyTorch optimizer API"""
        pass

    def _compute_loss_batched(self, batch_data, seeds_list, epsilon_sign):
        """
        Compute losses for N perturbations, processing in chunks to avoid OOM.

        Uses layer-wise materialization with chunking: processes perturbations in
        small batches (e.g., 8 at a time) to fit in GPU memory.
        """
        data_batch_size = batch_data.size(0)
        n_perts_total = len(seeds_list)
        chunk_size = self.perturbation_chunk_size

        all_losses = []

        # Process perturbations in chunks
        for chunk_start in range(0, n_perts_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_perts_total)
            seeds_chunk = seeds_list[chunk_start:chunk_end]
            n_perts = len(seeds_chunk)

            # Replicate data for this chunk: [n_perts * data_batch_size, seq_len]
            replicated_data = batch_data.repeat(n_perts, 1)

            with torch.no_grad():
                # Get all Linear layers
                linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]

                # Apply layerwise perturbations for this chunk
                # Each layer will materialize perturbations, compute, then free
                # This is done via nested context managers

                # Build nested context managers
                contexts = []
                for layer in linear_layers:
                    # Generate unique seed for this layer
                    layer_seeds = [s + hash(id(layer)) % 1000000 for s in seeds_chunk]
                    ctx = layerwise_perturbed_forward(layer, layer_seeds, epsilon_sign * self.epsilon)
                    contexts.append(ctx)

                # Enter all contexts
                for ctx in contexts:
                    ctx.__enter__()

                try:
                    # Forward pass with all layers perturbed
                    logits = self.model(replicated_data)
                finally:
                    # Exit all contexts (frees memory layer by layer)
                    for ctx in reversed(contexts):
                        ctx.__exit__(None, None, None)

                # Compute losses for each perturbation in this chunk
                targets = replicated_data[:, 1:]
                logits_shifted = logits[:, :-1]

                for i in range(n_perts):
                    start_idx = i * data_batch_size
                    end_idx = (i+1) * data_batch_size

                    loss = F.cross_entropy(
                        logits_shifted[start_idx:end_idx].reshape(-1, logits_shifted.size(-1)),
                        targets[start_idx:end_idx].reshape(-1),
                        reduction='mean'
                    )
                    all_losses.append(loss.item())

                # Free memory before next chunk
                del logits, logits_shifted, targets, replicated_data
                torch.cuda.empty_cache()

        return all_losses
    
    def step(self, loss_fn, batch_provider):
        """
        Optimization step with layer-wise materialization

        Args:
            loss_fn: Loss function (not used, we compute loss directly)
            batch_provider: Function that returns batches
        """
        start_time = time.time()

        # Get first batch
        # batch_provider returns (chunk, is_doc_end, actual_lengths)
        result = batch_provider()
        if isinstance(result, tuple):
            batch_data = result[0]  # Extract just the chunk tensor
        else:
            batch_data = result
        device = batch_data.device
        
        # Baseline loss
        with torch.no_grad():
            logits = self.model(batch_data)
            targets = batch_data[:, 1:]
            logits_shifted = logits[:, :-1]
            loss_baseline = F.cross_entropy(
                logits_shifted.reshape(-1, logits_shifted.size(-1)),
                targets.reshape(-1),
                reduction='mean'
            ).item()
        
        # Generate seeds for all perturbations
        seeds = [
            self.base_seed + self.step_counter * self.n_perturbations + i
            for i in range(self.n_perturbations)
        ]
        
        # Forward and backward perturbations (batched)
        losses_plus = self._compute_loss_batched(batch_data, seeds, +1.0)
        losses_minus = self._compute_loss_batched(batch_data, seeds, -1.0)
        
        # Estimate gradient
        grad_accumulator = torch.zeros(self.param_count, device=device)
        
        for i, seed in enumerate(seeds):
            grad_coef = (losses_plus[i] - losses_minus[i]) / (2 * self.epsilon)
            
            # Generate perturbation for gradient
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            pert_flat = torch.randn(
                self.param_count,
                generator=generator,
                device=device,
                dtype=torch.float32
            ).sign()
            
            grad_accumulator.add_(pert_flat, alpha=grad_coef)
        
        # Average gradient
        grad_accumulator.div_(self.n_perturbations)
        
        # Apply update
        offset = 0
        with torch.no_grad():
            for p in self.params:
                numel = p.numel()
                p_grad = grad_accumulator[offset:offset+numel].reshape(p.shape)
                p.data.add_(p_grad, alpha=-self.learning_rate)
                offset += numel
        
        self.step_counter += 1
        end_time = time.time()
        
        return {
            'loss': loss_baseline,
            'time_total': end_time - start_time,
            'step': self.step_counter,
        }

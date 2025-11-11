"""
Complete streaming cross-entropy with proper autograd support.

Forward: Compute loss without materializing full logits tensor
Backward: Compute gradients w.r.t. embeddings and lm_head weights
"""

import torch
import triton
import triton.language as tl


@triton.jit
def streaming_ce_forward_kernel(
    # Input pointers
    embed_ptr,        # [batch, seq, dim]
    weight_ptr,       # [vocab, dim]
    labels_ptr,       # [batch, seq]
    # Output pointers
    loss_ptr,         # [batch, seq]
    valid_ptr,        # [batch, seq]
    # Shapes
    batch_size, seq_len, dim, vocab_size,
    # Strides
    embed_batch_stride, embed_seq_stride, embed_dim_stride,
    weight_vocab_stride, weight_dim_stride,
    labels_batch_stride, labels_seq_stride,
    loss_batch_stride, loss_seq_stride,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
):
    """Compute cross-entropy loss for one (batch, seq) position."""
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Load label
    label_offset = batch_idx * labels_batch_stride + seq_idx * labels_seq_stride
    label = tl.load(labels_ptr + label_offset)

    # Check validity
    is_valid = label != -100
    valid_offset = batch_idx * loss_batch_stride + seq_idx * loss_seq_stride
    tl.store(valid_ptr + valid_offset, is_valid.to(tl.float32))

    if not is_valid:
        tl.store(loss_ptr + valid_offset, 0.0)
        return

    # Load embedding [BLOCK_DIM]
    embed_offset = batch_idx * embed_batch_stride + seq_idx * embed_seq_stride
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < dim
    embed_vec = tl.load(
        embed_ptr + embed_offset + dim_offsets * embed_dim_stride,
        mask=dim_mask,
        other=0.0
    )

    # Two-pass: find max, then compute logsumexp
    max_logit = float('-inf')
    num_vocab_blocks = tl.cdiv(vocab_size, VOCAB_BLOCK)

    # Pass 1: Find max
    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        # Load weights [VOCAB_BLOCK, BLOCK_DIM]
        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        # Compute logits [VOCAB_BLOCK]
        logits = tl.sum(weight_block * embed_vec[None, :], axis=1)
        block_max = tl.max(tl.where(vocab_mask, logits, float('-inf')))
        max_logit = tl.maximum(max_logit, block_max)

    # Pass 2: Compute logsumexp and extract target
    sum_exp = 0.0
    target_logit = 0.0

    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        logits = tl.sum(weight_block * embed_vec[None, :], axis=1)

        # Accumulate exp sum
        exp_block = tl.exp(logits - max_logit)
        sum_exp += tl.sum(tl.where(vocab_mask, exp_block, 0.0))

        # Extract target logit
        is_target = (vocab_offsets == label) & vocab_mask
        target_logit += tl.sum(tl.where(is_target, logits, 0.0))

    # Compute loss
    log_sum_exp = tl.log(sum_exp) + max_logit
    loss = -(target_logit - log_sum_exp)
    tl.store(loss_ptr + valid_offset, loss)


@triton.jit
def streaming_ce_backward_kernel(
    # Input pointers
    embed_ptr,           # [batch, seq, dim]
    weight_ptr,          # [vocab, dim]
    labels_ptr,          # [batch, seq]
    grad_output_ptr,     # scalar (1.0 / num_valid)
    # Output pointers
    grad_embed_ptr,      # [batch, seq, dim]
    grad_weight_ptr,     # [vocab, dim] - atomic adds
    # Shapes
    batch_size, seq_len, dim, vocab_size,
    # Strides
    embed_batch_stride, embed_seq_stride, embed_dim_stride,
    weight_vocab_stride, weight_dim_stride,
    labels_batch_stride, labels_seq_stride,
    grad_embed_batch_stride, grad_embed_seq_stride, grad_embed_dim_stride,
    grad_weight_vocab_stride, grad_weight_dim_stride,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
):
    """
    Compute gradients for cross-entropy.

    For each valid token:
    - grad_embed = weight^T @ (softmax - one_hot)
    - grad_weight[label] -= embed (accumulated across all tokens)
    - grad_weight[other] += softmax[other] * embed
    """
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Load label
    label_offset = batch_idx * labels_batch_stride + seq_idx * labels_seq_stride
    label = tl.load(labels_ptr + label_offset)

    if label == -100:
        return  # Skip invalid tokens

    # Load embedding
    embed_offset = batch_idx * embed_batch_stride + seq_idx * embed_seq_stride
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < dim
    embed_vec = tl.load(
        embed_ptr + embed_offset + dim_offsets * embed_dim_stride,
        mask=dim_mask,
        other=0.0
    )

    # Load grad_output (scalar: 1 / num_valid)
    grad_out = tl.load(grad_output_ptr)

    # Recompute forward pass to get logits and max
    max_logit = float('-inf')
    num_vocab_blocks = tl.cdiv(vocab_size, VOCAB_BLOCK)

    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        logits = tl.sum(weight_block * embed_vec[None, :], axis=1)
        block_max = tl.max(tl.where(vocab_mask, logits, float('-inf')))
        max_logit = tl.maximum(max_logit, block_max)

    # Compute logsumexp for softmax
    sum_exp = 0.0
    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        logits = tl.sum(weight_block * embed_vec[None, :], axis=1)
        exp_block = tl.exp(logits - max_logit)
        sum_exp += tl.sum(tl.where(vocab_mask, exp_block, 0.0))

    # Now compute gradients: softmax - one_hot
    grad_embed_accum = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        logits = tl.sum(weight_block * embed_vec[None, :], axis=1)

        # Softmax probabilities
        softmax_probs = tl.exp(logits - max_logit) / sum_exp

        # Subtract one-hot for target
        is_target = (vocab_offsets == label) & vocab_mask
        grad_logits = tl.where(is_target, softmax_probs - 1.0, softmax_probs)
        grad_logits = tl.where(vocab_mask, grad_logits * grad_out, 0.0)

        # grad_embed += weight^T @ grad_logits
        grad_embed_accum += tl.sum(weight_block * grad_logits[:, None], axis=0)

        # grad_weight atomically (each vocab item gets gradient from this token)
        for i in range(VOCAB_BLOCK):
            if vocab_mask[i]:
                vocab_idx = vocab_offsets[i]
                grad_w_scale = grad_logits[i]

                # Atomic add to grad_weight[vocab_idx, :]
                for d in range(BLOCK_DIM):
                    if dim_mask[d]:
                        grad_w_offset = vocab_idx * grad_weight_vocab_stride + d * grad_weight_dim_stride
                        tl.atomic_add(grad_weight_ptr + grad_w_offset, embed_vec[d] * grad_w_scale)

    # Store grad_embed
    grad_embed_offset = batch_idx * grad_embed_batch_stride + seq_idx * grad_embed_seq_stride
    tl.store(
        grad_embed_ptr + grad_embed_offset + dim_offsets * grad_embed_dim_stride,
        grad_embed_accum,
        mask=dim_mask
    )


class StreamingCrossEntropyFunction(torch.autograd.Function):
    """Autograd function wrapper for streaming cross-entropy."""

    @staticmethod
    def forward(ctx, embeddings, lm_head_weight, labels):
        """
        Args:
            embeddings: [batch, seq, dim]
            lm_head_weight: [vocab, dim]
            labels: [batch, seq]
        Returns:
            loss: scalar
        """
        batch_size, seq_len, dim = embeddings.shape
        vocab_size = lm_head_weight.shape[0]

        # Allocate output
        loss_per_token = torch.zeros(batch_size, seq_len, device=embeddings.device, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, seq_len, device=embeddings.device, dtype=torch.float32)

        # Launch forward kernel
        grid = (batch_size * seq_len,)
        BLOCK_DIM = triton.next_power_of_2(dim)
        VOCAB_BLOCK = 128

        streaming_ce_forward_kernel[grid](
            embeddings, lm_head_weight, labels,
            loss_per_token, valid_mask,
            batch_size, seq_len, dim, vocab_size,
            embeddings.stride(0), embeddings.stride(1), embeddings.stride(2),
            lm_head_weight.stride(0), lm_head_weight.stride(1),
            labels.stride(0), labels.stride(1),
            loss_per_token.stride(0), loss_per_token.stride(1),
            BLOCK_DIM=BLOCK_DIM,
            VOCAB_BLOCK=VOCAB_BLOCK,
        )

        # Compute mean loss
        total_loss = (loss_per_token * valid_mask).sum()
        total_valid = valid_mask.sum().clamp(min=1)
        loss = total_loss / total_valid

        # Save for backward
        ctx.save_for_backward(embeddings, lm_head_weight, labels)
        ctx.grad_output_scale = 1.0 / total_valid.item()
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.dim = dim
        ctx.vocab_size = vocab_size
        ctx.BLOCK_DIM = BLOCK_DIM
        ctx.VOCAB_BLOCK = VOCAB_BLOCK

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        """
        Args:
            grad_loss: scalar gradient from upstream
        Returns:
            grad_embeddings, grad_lm_head_weight, None (for labels)
        """
        embeddings, lm_head_weight, labels = ctx.saved_tensors

        # Allocate gradient tensors
        grad_embeddings = torch.zeros_like(embeddings)
        grad_lm_head_weight = torch.zeros_like(lm_head_weight)

        # grad_output scalar: grad_loss / num_valid (already scaled in forward)
        grad_output_tensor = torch.tensor([grad_loss.item() * ctx.grad_output_scale],
                                          device=embeddings.device, dtype=torch.float32)

        # Launch backward kernel
        grid = (ctx.batch_size * ctx.seq_len,)

        streaming_ce_backward_kernel[grid](
            embeddings, lm_head_weight, labels, grad_output_tensor,
            grad_embeddings, grad_lm_head_weight,
            ctx.batch_size, ctx.seq_len, ctx.dim, ctx.vocab_size,
            embeddings.stride(0), embeddings.stride(1), embeddings.stride(2),
            lm_head_weight.stride(0), lm_head_weight.stride(1),
            labels.stride(0), labels.stride(1),
            grad_embeddings.stride(0), grad_embeddings.stride(1), grad_embeddings.stride(2),
            grad_lm_head_weight.stride(0), grad_lm_head_weight.stride(1),
            BLOCK_DIM=ctx.BLOCK_DIM,
            VOCAB_BLOCK=ctx.VOCAB_BLOCK,
        )

        return grad_embeddings, grad_lm_head_weight, None


def triton_streaming_cross_entropy(embeddings, lm_head_weight, labels):
    """
    Complete streaming cross-entropy with autograd support.

    NO LOGITS MATERIALIZATION in forward or backward!
    """
    return StreamingCrossEntropyFunction.apply(embeddings, lm_head_weight, labels)


__all__ = ['triton_streaming_cross_entropy']

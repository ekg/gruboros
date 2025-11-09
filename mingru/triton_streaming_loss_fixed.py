"""
Fixed Triton kernel for streaming cross-entropy loss computation.

NO LOGITS MATERIALIZATION!
Computes loss token-by-token directly from embeddings.

Memory: O(batch * vocab) instead of O(batch * seq * vocab)

FIX: Replaced Python for loops with proper Triton vectorization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def streaming_ce_loss_kernel(
    # Pointers
    embed_ptr,        # [batch, seq, dim]
    weight_ptr,       # [vocab, dim] - lm_head weight (transposed!)
    labels_ptr,       # [batch, seq]
    loss_ptr,         # [batch, seq] - output loss per token
    valid_ptr,        # [batch, seq] - output valid mask
    # Shapes
    batch_size,
    seq_len,
    dim,
    vocab_size,
    # Strides
    embed_batch_stride,
    embed_seq_stride,
    embed_dim_stride,
    weight_vocab_stride,
    weight_dim_stride,
    labels_batch_stride,
    labels_seq_stride,
    loss_batch_stride,
    loss_seq_stride,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
):
    """
    Compute cross-entropy loss for one token WITHOUT materializing logits.

    Strategy: Each program handles one (batch, seq) position.
    Process vocabulary in VOCAB_BLOCK chunks, computing logits on-the-fly.
    """
    # Program ID for (batch, seq) position
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Load label for this position
    label_offset = batch_idx * labels_batch_stride + seq_idx * labels_seq_stride
    label = tl.load(labels_ptr + label_offset)

    # Check if valid (not -100 ignore index)
    is_valid = label != -100

    # Store validity
    valid_offset = batch_idx * loss_batch_stride + seq_idx * loss_seq_stride
    tl.store(valid_ptr + valid_offset, is_valid.to(tl.float32))

    if not is_valid:
        tl.store(loss_ptr + valid_offset, 0.0)
        return

    # Load embedding for this (batch, seq) position [dim]
    embed_offset = batch_idx * embed_batch_stride + seq_idx * embed_seq_stride

    # Load full embedding vector [BLOCK_SIZE] at a time
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offsets < dim
    embed_vec = tl.load(
        embed_ptr + embed_offset + dim_offsets * embed_dim_stride,
        mask=dim_mask,
        other=0.0
    )

    # Process vocabulary in chunks of VOCAB_BLOCK
    # We'll compute logits for VOCAB_BLOCK vocab items at a time

    # First pass: find max logit (for numerical stability)
    max_logit = float('-inf')

    # Number of vocab blocks to process
    num_vocab_blocks = tl.cdiv(vocab_size, VOCAB_BLOCK)

    # Loop over vocab blocks (compile-time unrolled for small num_vocab_blocks)
    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        # Compute logits for this vocab block: [VOCAB_BLOCK]
        # logit[v] = dot(embed, weight[v, :])

        # Load weight matrix for this vocab block [VOCAB_BLOCK, BLOCK_SIZE]
        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        # Compute dot product: [VOCAB_BLOCK] = [VOCAB_BLOCK, BLOCK_SIZE] @ [BLOCK_SIZE]
        logits_block = tl.sum(weight_block * embed_vec[None, :], axis=1)

        # Update max (only consider valid vocab indices)
        block_max = tl.max(tl.where(vocab_mask, logits_block, float('-inf')))
        max_logit = tl.maximum(max_logit, block_max)

    # Second pass: compute logsumexp and extract target logit
    sum_exp = 0.0
    target_logit = 0.0

    for vocab_block_idx in range(num_vocab_blocks):
        vocab_start = vocab_block_idx * VOCAB_BLOCK
        vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
        vocab_mask = vocab_offsets < vocab_size

        # Recompute logits for this vocab block
        weight_offsets = vocab_offsets[:, None] * weight_vocab_stride + dim_offsets[None, :] * weight_dim_stride
        weight_block = tl.load(
            weight_ptr + weight_offsets,
            mask=vocab_mask[:, None] & dim_mask[None, :],
            other=0.0
        )

        logits_block = tl.sum(weight_block * embed_vec[None, :], axis=1)

        # Compute exp(logit - max) and accumulate
        exp_block = tl.exp(logits_block - max_logit)
        sum_exp += tl.sum(tl.where(vocab_mask, exp_block, 0.0))

        # Check if target label is in this block
        is_target = (vocab_offsets == label) & vocab_mask
        target_logit += tl.sum(tl.where(is_target, logits_block, 0.0))

    # Compute loss: -log_prob = -(logit - logsumexp)
    log_sum_exp = tl.log(sum_exp) + max_logit
    loss = -(target_logit - log_sum_exp)

    # Store loss
    tl.store(loss_ptr + valid_offset, loss)


def triton_streaming_cross_entropy(embeddings, lm_head_weight, labels):
    """
    Compute cross-entropy loss using Triton kernel.

    NO LOGITS MATERIALIZATION!

    Args:
        embeddings: [batch, seq, dim] - final layer embeddings
        lm_head_weight: [vocab, dim] - lm_head.weight
        labels: [batch, seq] - target tokens

    Returns:
        Scalar loss (averaged over valid tokens)
    """
    batch_size, seq_len, dim = embeddings.shape
    vocab_size = lm_head_weight.shape[0]

    # Output tensors
    loss_per_token = torch.zeros(batch_size, seq_len, device=embeddings.device, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, seq_len, device=embeddings.device, dtype=torch.float32)

    # Launch kernel: one program per (batch, seq) position
    grid = (batch_size * seq_len,)

    # Block sizes - must match embedding dim exactly or use next power of 2
    BLOCK_SIZE = triton.next_power_of_2(dim)
    VOCAB_BLOCK = 128  # Process vocab in chunks

    streaming_ce_loss_kernel[grid](
        embeddings, lm_head_weight, labels,
        loss_per_token, valid_mask,
        batch_size, seq_len, dim, vocab_size,
        embeddings.stride(0), embeddings.stride(1), embeddings.stride(2),
        lm_head_weight.stride(0), lm_head_weight.stride(1),
        labels.stride(0), labels.stride(1),
        loss_per_token.stride(0), loss_per_token.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        VOCAB_BLOCK=VOCAB_BLOCK,
    )

    # Sum losses and count valid tokens
    total_loss = (loss_per_token * valid_mask).sum()
    total_valid = valid_mask.sum()

    return total_loss / total_valid.clamp(min=1)


# Export
__all__ = ['triton_streaming_cross_entropy']

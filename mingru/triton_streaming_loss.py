"""
Triton kernel for streaming cross-entropy loss computation.

NO LOGITS MATERIALIZATION!
Computes loss token-by-token directly from embeddings.

Memory: O(batch * vocab) instead of O(batch * seq * vocab)
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

    For each (batch, seq) position:
    1. Load embedding [dim]
    2. Compute logits on-the-fly by matmul with weight [vocab, dim]
    3. Compute logsumexp across vocab
    4. Extract log_prob for target label
    5. Loss = -log_prob
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
    embed_base_ptr = embed_ptr + embed_offset

    # We'll compute logits in blocks across vocabulary
    # For numerical stability, compute max first, then logsumexp

    # PASS 1: Find max logit for numerical stability
    max_logit = float('-inf')

    for vocab_start in range(0, vocab_size, VOCAB_BLOCK):
        vocab_end = min(vocab_start + VOCAB_BLOCK, vocab_size)
        vocab_offsets = tl.arange(0, VOCAB_BLOCK) + vocab_start
        vocab_mask = vocab_offsets < vocab_size

        # Compute logits for this vocab block
        # logits[v] = sum_d embed[d] * weight[v, d]
        logits_block = tl.zeros([VOCAB_BLOCK], dtype=tl.float32)

        for dim_start in range(0, dim, BLOCK_SIZE):
            dim_offsets = tl.arange(0, BLOCK_SIZE) + dim_start
            dim_mask = dim_offsets < dim

            # Load embedding chunk [BLOCK_SIZE]
            embed_chunk = tl.load(
                embed_base_ptr + dim_offsets * embed_dim_stride,
                mask=dim_mask,
                other=0.0
            )

            # Load weight chunk [VOCAB_BLOCK, BLOCK_SIZE]
            # For each vocab in block, load its embedding
            for i in range(VOCAB_BLOCK):
                if vocab_offsets[i] < vocab_size:
                    weight_row_ptr = weight_ptr + vocab_offsets[i] * weight_vocab_stride
                    weight_chunk = tl.load(
                        weight_row_ptr + dim_offsets * weight_dim_stride,
                        mask=dim_mask,
                        other=0.0
                    )
                    # Accumulate dot product
                    logits_block = tl.where(
                        i == tl.arange(0, VOCAB_BLOCK),
                        logits_block + tl.sum(embed_chunk * weight_chunk),
                        logits_block
                    )

        # Update max
        block_max = tl.max(tl.where(vocab_mask, logits_block, float('-inf')))
        max_logit = tl.maximum(max_logit, block_max)

    # PASS 2: Compute logsumexp using the max
    sum_exp = 0.0
    target_logit = 0.0

    for vocab_start in range(0, vocab_size, VOCAB_BLOCK):
        vocab_end = min(vocab_start + VOCAB_BLOCK, vocab_size)
        vocab_offsets = tl.arange(0, VOCAB_BLOCK) + vocab_start
        vocab_mask = vocab_offsets < vocab_size

        # Recompute logits for this vocab block (same as above)
        logits_block = tl.zeros([VOCAB_BLOCK], dtype=tl.float32)

        for dim_start in range(0, dim, BLOCK_SIZE):
            dim_offsets = tl.arange(0, BLOCK_SIZE) + dim_start
            dim_mask = dim_offsets < dim

            embed_chunk = tl.load(
                embed_base_ptr + dim_offsets * embed_dim_stride,
                mask=dim_mask,
                other=0.0
            )

            for i in range(VOCAB_BLOCK):
                if vocab_offsets[i] < vocab_size:
                    weight_row_ptr = weight_ptr + vocab_offsets[i] * weight_vocab_stride
                    weight_chunk = tl.load(
                        weight_row_ptr + dim_offsets * weight_dim_stride,
                        mask=dim_mask,
                        other=0.0
                    )
                    logits_block = tl.where(
                        i == tl.arange(0, VOCAB_BLOCK),
                        logits_block + tl.sum(embed_chunk * weight_chunk),
                        logits_block
                    )

        # Compute exp(logit - max) and accumulate
        exp_block = tl.exp(logits_block - max_logit)
        sum_exp += tl.sum(tl.where(vocab_mask, exp_block, 0.0))

        # Check if target label is in this block
        if vocab_start <= label < vocab_end:
            label_idx_in_block = label - vocab_start
            target_logit = logits_block[label_idx_in_block]

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

    BLOCK_SIZE = 64  # For dim reduction
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


def streaming_cross_entropy_loss_simple(embeddings, lm_head, labels):
    """
    Simple fallback: compute logits one token at a time.

    Still way better than materializing full [batch, seq, vocab]!

    Args:
        embeddings: [batch, seq, dim]
        lm_head: nn.Linear layer
        labels: [batch, seq]

    Returns:
        Scalar loss
    """
    batch_size, seq_len, dim = embeddings.shape

    total_loss = 0.0
    total_count = 0

    # Process one position at a time
    for pos in range(seq_len):
        # Extract embeddings for this position [batch, dim]
        embed_pos = embeddings[:, pos, :]  # [batch, dim]

        # Compute logits for this position only [batch, vocab]
        logits_pos = lm_head(embed_pos)  # [batch, vocab]

        # Get labels for this position [batch]
        labels_pos = labels[:, pos]

        # Compute loss (ignore -100)
        valid_mask = labels_pos != -100
        if valid_mask.sum() > 0:
            loss = torch.nn.functional.cross_entropy(
                logits_pos[valid_mask],
                labels_pos[valid_mask],
                reduction='sum'
            )
            total_loss += loss
            total_count += valid_mask.sum().item()

    return total_loss / max(total_count, 1)


# Export both versions
__all__ = ['triton_streaming_cross_entropy', 'streaming_cross_entropy_loss_simple']

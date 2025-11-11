"""
Triton kernel for streaming cross-entropy loss.

Computes loss position-by-position WITHOUT materializing logits.
Each thread handles one position: computes logsumexp(W @ h) - W[target] @ h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def streaming_ce_loss_kernel(
    hidden_ptr,      # [batch*seq, dim]
    weight_ptr,      # [vocab, dim]
    targets_ptr,     # [batch*seq]
    loss_ptr,        # [batch*seq] output
    valid_ptr,       # [batch*seq] mask
    batch_seq: tl.constexpr,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Compute cross-entropy loss for one position.

    Each program handles one (batch, seq) position:
    1. Load hidden vector [dim]
    2. Compute dot products with ALL vocab embeddings [vocab, dim]
    3. Compute logsumexp across vocab
    4. Compute target score
    5. Return log_partition - target_score
    """
    pos_idx = tl.program_id(0)

    if pos_idx >= batch_seq:
        return

    # Load target
    target = tl.load(targets_ptr + pos_idx)

    # Check if ignored
    if target == ignore_index:
        tl.store(valid_ptr + pos_idx, 0)
        tl.store(loss_ptr + pos_idx, 0.0)
        return

    tl.store(valid_ptr + pos_idx, 1)

    # Load hidden vector [dim]
    hidden_offset = pos_idx * hidden_dim
    hidden = tl.load(hidden_ptr + hidden_offset + tl.arange(0, hidden_dim))

    # Compute logits = W @ h for all vocab entries
    # We'll do this in blocks to avoid register pressure

    max_val = float('-inf')

    # First pass: find max for numerical stability
    for vocab_start in range(0, vocab_size, BLOCK_SIZE):
        vocab_end = min(vocab_start + BLOCK_SIZE, vocab_size)
        vocab_range = vocab_end - vocab_start

        # Load weight block [BLOCK_SIZE, dim]
        # Compute dot products
        logits_block = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for d in range(0, hidden_dim):
            h_val = tl.load(hidden_ptr + hidden_offset + d)
            w_block = tl.load(
                weight_ptr + (vocab_start + tl.arange(0, BLOCK_SIZE)) * hidden_dim + d,
                mask=tl.arange(0, BLOCK_SIZE) < vocab_range
            )
            logits_block += h_val * w_block

        block_max = tl.max(logits_block, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute logsumexp
    sum_exp = 0.0
    target_score = 0.0

    for vocab_start in range(0, vocab_size, BLOCK_SIZE):
        vocab_end = min(vocab_start + BLOCK_SIZE, vocab_size)
        vocab_range = vocab_end - vocab_start

        # Recompute logits for this block
        logits_block = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for d in range(0, hidden_dim):
            h_val = tl.load(hidden_ptr + hidden_offset + d)
            w_block = tl.load(
                weight_ptr + (vocab_start + tl.arange(0, BLOCK_SIZE)) * hidden_dim + d,
                mask=tl.arange(0, BLOCK_SIZE) < vocab_range
            )
            logits_block += h_val * w_block

        # Accumulate exp(logits - max)
        exp_block = tl.exp(logits_block - max_val)
        sum_exp += tl.sum(exp_block, axis=0)

        # Check if target is in this block
        vocab_indices = vocab_start + tl.arange(0, BLOCK_SIZE)
        is_target = vocab_indices == target
        target_score += tl.sum(tl.where(is_target, logits_block, 0.0))

    # Compute final loss
    log_partition = max_val + tl.log(sum_exp)
    loss = log_partition - target_score

    tl.store(loss_ptr + pos_idx, loss)


def triton_streaming_ce_loss(
    hidden_states: torch.Tensor,  # [batch, seq, dim]
    weight: torch.Tensor,          # [vocab, dim]
    targets: torch.Tensor,         # [batch, seq]
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Triton-based streaming cross-entropy loss.

    Uses kernel that computes loss position-by-position WITHOUT materializing logits.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    vocab_size = weight.shape[0]

    batch_seq = batch_size * seq_len

    # Flatten inputs
    hidden_flat = hidden_states.reshape(batch_seq, hidden_dim).contiguous()
    targets_flat = targets.reshape(batch_seq).contiguous()

    # Output buffers
    loss_out = torch.zeros(batch_seq, device=hidden_states.device, dtype=torch.float32)
    valid_out = torch.zeros(batch_seq, device=hidden_states.device, dtype=torch.int32)

    # Launch kernel (one thread per position)
    grid = (batch_seq,)

    streaming_ce_loss_kernel[grid](
        hidden_flat,
        weight,
        targets_flat,
        loss_out,
        valid_out,
        batch_seq,
        hidden_dim,
        vocab_size,
        ignore_index,
        BLOCK_SIZE=128,
    )

    # Average over valid positions
    num_valid = valid_out.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=hidden_states.device)

    return loss_out.sum() / num_valid.float()

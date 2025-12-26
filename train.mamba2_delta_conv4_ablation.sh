#!/bin/bash
set -e -x

# =============================================================================
# Mamba2-Style Delta + Conv4 Ablation
# =============================================================================
# Testing Mamba2-style delta (softplus/exp) WITH conv4 local context.
# Mamba2 uses a 1D causal conv (kernel=4) before the SSM for local features.
#
# Architecture:
#   x_conv = CausalConv1d(x, kernel=4)  # Local context
#   candidate = tanh(R @ h + Wx @ x_conv + b)
#   delta = softplus(W_delta @ x + b_delta)
#   decay = exp(-delta)
#   h_new = decay * h + (1 - decay) * candidate
#   gate = group_softmax(W1 @ x) * silu(W2 @ x)
#   output = W_out @ norm(h_new * gate)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mamba2_delta_conv4"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29576

echo "======================================================================="
echo "=== Mamba2-Style Delta + Conv4 (local context) ==="
echo "======================================================================="
echo "Architecture: CausalConv4 -> R@h + tanh + Mamba2 delta + compete×silu"
echo "Testing if conv4 adds value on top of mamba2_delta"
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/torchrun \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  --dim 2048 \
  --depth 32 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_leaky_mamba2_delta \
  --conv_kernel_size 4 \
  --compete_n_groups 32 \
  --recurrence_chunk_size 64 \
  --delta_init -1.8 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 300 \
  --lr 0.0006 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 100 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/mamba2_delta_conv4_${TIMESTAMP}.log"

echo "Done!"

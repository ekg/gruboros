#!/bin/bash
set -e -x

# =============================================================================
# LeakyElman 1B Training - Leaky integration RNN with input-only output gate
# =============================================================================
# Architecture:
#   - candidate = tanh(R @ h + Wx @ x + b)       -- Elman candidate
#   - dt = softplus(W_delta @ x + b_delta)       -- input-dependent timestep
#   - decay_rate = exp(-exp(A_log))              -- LOG-SPACE: always in (0,1)!
#   - alpha = exp(-dt * decay_rate)              -- blend factor
#   - h_new = alpha * h + (1 - alpha) * candidate -- leaky integration
#   - gate = silu(W_gate @ x + b)                -- INPUT-ONLY output gate
#   - output = h_new * gate
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_leaky_elman_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29574

echo "======================================================================="
echo "=== LeakyElman 1B (Leaky Integration + INPUT-ONLY Output Gate) ==="
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
  --use_leaky_elman \
  --delta_init 3.0 \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 3000 \
  --lr 0.0001 \
  --weight_decay 0.1 \
  --grad_clip 0.1 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/leaky_elman_1b_${TIMESTAMP}.log"

echo "Done!"

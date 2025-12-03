#!/bin/bash
set -e -x

# =============================================================================
# PyTorch cuDNN GRU + Multi-Scale EMA: CUMULATIVE Experiment 3 (ff4 + multiscale)
# =============================================================================
# Builds on ff4: ff_mult=4, depth=11 for ~1B params (3 EMA tracks add params)
# PLUS: 3 EMA tracks at different timescales per layer:
# - Fast EMA (α ~ 0.1): captures short-range patterns (~7 token half-life)
# - Medium EMA (α ~ 0.01): captures medium-range patterns (~70 token half-life)
# - Slow EMA (α ~ 0.001): captures long-range patterns (~700 token half-life)
# Each EMA track has its own learnable alpha initialized at these values.
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_ema_multiscale"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510

echo "======================================================================="
echo "=== PyTorch cuDNN GRU + Multi-Scale EMA (3 timescales) Training ==="
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
  --depth 11 \
  --expansion_factor 1.0 \
  --ff_mult 4.0 \
  --dropout 0.0 \
  \
  --use_cudnn_multiscale_ema_gru \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 10000 \
  --lr 0.001 \
  --weight_decay 0.033 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/cudnn_ema_multiscale_${TIMESTAMP}.log"

echo "Done!"

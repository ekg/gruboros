#!/bin/bash
set -e -x

# =============================================================================
# PyTorch cuDNN GRU + EMA: CUMULATIVE Experiment 1 (ff4 baseline)
# =============================================================================
# Baseline for cumulative experiments: ff_mult=4, depth=14 for ~1B params
# Adding feedforward network to see if per-token transformation
# capacity improves language modeling.
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_ema_ff4"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510

echo "======================================================================="
echo "=== PyTorch cuDNN GRU+EMA 1B + FF_MULT=4 Training ==="
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
  --depth 14 \
  --expansion_factor 1.0 \
  --ff_mult 4.0 \
  --dropout 0.0 \
  \
  --use_cudnn_ema_gru \
  --ema_alpha 0.01 \
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
  2>&1 | tee "logs/cudnn_ema_ff4_${TIMESTAMP}.log"

echo "Done!"

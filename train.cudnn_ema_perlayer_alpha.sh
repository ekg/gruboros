#!/bin/bash
set -e -x

# =============================================================================
# PyTorch cuDNN GRU + EMA Training with Per-Layer Alpha Initialization
# =============================================================================
# Test: Initialize each layer with different EMA alpha values:
# - Early layers: fast (high α ~ 0.05) - capture local patterns
# - Deep layers: slow (low α ~ 0.005) - capture long-range dependencies
# Formula: init_alpha = 0.05 * (0.1 ** (layer_idx / (num_layers - 1)))
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_ema_perlayer_alpha"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510

echo "======================================================================="
echo "=== PyTorch cuDNN GRU+EMA 1B + Per-Layer Alpha Training ==="
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
  --depth 24 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_cudnn_ema_gru \
  --per_layer_alpha \
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
  2>&1 | tee "logs/cudnn_ema_perlayer_alpha_${TIMESTAMP}.log"

echo "Done!"

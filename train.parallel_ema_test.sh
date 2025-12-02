#!/bin/bash
set -e -x

# =============================================================================
# PARALLEL EMA GRU 1B TEST - Triton-based (DDP-compatible)
# =============================================================================
# Uses parallel scan for EMA + Triton kernel for GRU cell
# ~1B params: dim=2048, depth=23, ff_mult=0, expansion=1.0
# Fallback from FlashRNN (which has DDP issues)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_parallel_ema_test"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
NUM_GPUS=8

echo "======================================================================="
echo "=== PARALLEL EMA GRU TEST - 10x faster! ==="
echo "======================================================================="
echo "Architecture: ParallelEMA_GRU (parallel scan EMA + Triton GRU)"
echo "EMA alpha=0.01 (half-life ~70 tokens)"
echo "Expected: ~100k+ tokens/sec (vs 15k sequential)"
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
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
  --depth 23 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_ema_gru \
  --ema_alpha 0.01 \
  --z_bias_input 0.0 \
  --z_bias_hidden 0.0 \
  \
  --chunk_size 512 \
  --batch_size 8 \
  --grad_accum 32 \
  \
  --train_steps 1000 \
  --lr 0.001 \
  --sgd \
  --momentum 0.0 \
  --weight_decay 0.033 \
  --grad_clip 0.0 \
  \
  --save_every 500 \
  --keep_checkpoints 10 \
  --keep_elite 32 \
  --milestone_every 10000 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "Done!"

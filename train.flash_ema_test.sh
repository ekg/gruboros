#!/bin/bash
set -e -x

# =============================================================================
# FLASH EMA GRU TEST - 60x faster than sequential!
# =============================================================================
# Uses FlashRNN for GRU (hardware-optimized) + parallel EMA for long-range memory
# Expected throughput: ~900k+ tok/s at scale
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_flash_ema_test"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
NUM_GPUS=8

# Add ninja to PATH for FlashRNN JIT compilation
export PATH=/home/erikg/micromamba/envs/mingru/bin:$PATH

echo "======================================================================="
echo "=== FLASH EMA GRU TEST - FlashRNN + Parallel EMA ==="
echo "======================================================================="
echo "Architecture: FlashGRU_EMA (FlashRNN GRU + parallel EMA)"
echo "EMA alpha=0.01 (half-life ~70 tokens)"
echo "Expected: ~900k tokens/sec (60x faster than sequential EMA GRU)"
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
  --use_flash_ema_gru \
  --ema_alpha 0.01 \
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

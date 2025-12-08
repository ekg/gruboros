#!/bin/bash
set -e -x

# =============================================================================
# Mamba2 SSM Training - DDP mode with 8 GPUs
# =============================================================================
# Model: ~1B params using Mamba2 with SSD (State Space Duality)
# Mamba2 uses larger d_state (64) and headdim parameter for faster training
# Note: Mamba2 doesn't use TBPTT - processes full chunk_size as context
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mamba2_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29521

echo "======================================================================="
echo "=== Mamba2 SSM 1B Training (SSD - faster parallel training) ==="
echo "======================================================================="

# Mamba2 1B config: dim=2048, depth=35, d_state=64, expand=2, headdim=64
# depth=35 gives ~1.0B params

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
  --depth 35 \
  --dropout 0.0 \
  \
  --use_mamba2 \
  --mamba_d_state 64 \
  --mamba_expand 2 \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 10000 \
  --lr 0.0006 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/mamba2_1b_${TIMESTAMP}.log"

echo "Done!"

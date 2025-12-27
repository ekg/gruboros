#!/bin/bash
set -e -x

# =============================================================================
# Diagonal MHTR DEEP - ~1.3B params (dim=1920, depth=48)
# =============================================================================
# KEY CHANGE: Diagonal (element-wise) transitions instead of full R matrix
# This should be stable at depth=48 like Mamba2!
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_diagonal_mhtr_deep_1.3b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29506
export PYTHONPATH=/home/erikg/haste_src:$PYTHONPATH

echo "======================================================================="
echo "=== Diagonal MHTR DEEP (dim=1920, depth=48, ~1.3B) ==="
echo "=== DIAGONAL transitions for depth stability ==="
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
  --dim 1920 \
  --depth 48 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_diagonal_mhtr \
  --mhtr_expand 2 \
  --mhtr_headdim 64 \
  --delta_init -2.0 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 12 \
  --grad_accum 1 \
  \
  --train_steps 1000 \
  --lr 0.0005 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 200 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/diagonal_mhtr_deep_1.3b_${TIMESTAMP}.log"

echo "Done!"

#!/bin/bash
set -e -x

# =============================================================================
# MultiHeadElman 1B Training - 2048x more expressive recurrence than Mamba2!
# =============================================================================
# Model: ~1.0B params using multi-head RNN with per-head R matrices
# Architecture:
#   - 64 heads × 64×64 R matrices = 262K recurrence params per layer
#   - Softsign activation (gradient-friendly, non-saturating)
#   - Input-only output gate (like Mamba2)
#   - depth=31 to match Mamba2's ~1B param count
#
# Key insight: Mamba2 uses 64 scalar decays (one per head).
# MultiHeadElman uses 64 heads × 64×64 matrices = 4096x more expressive!
#
# Comparison:
#   - Mamba2 recurrence params: 64 (scalars)
#   - MultiHeadElman recurrence params: 262,144 (per layer)
#   - Full Elman recurrence params: 4,194,304 (too large to train)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_multihead_elman_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29550

echo "======================================================================="
echo "=== MultiHeadElman 1B Training (64 heads × 64×64, softsign) ==="
echo "======================================================================="

# Config: dim=2048, depth=31, nheads=64, headdim=64 → 1.003B params
# Close to Mamba2's depth=35, matched on param count

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
  --depth 31 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_multihead_elman \
  --multihead_elman_nheads 64 \
  --multihead_elman_headdim 64 \
  --multihead_elman_activation softsign \
  --recurrence_chunk_size 64 \
  --no-tbptt \
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
  2>&1 | tee "logs/multihead_elman_1b_${TIMESTAMP}.log"

echo "Done!"

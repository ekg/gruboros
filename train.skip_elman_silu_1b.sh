#!/bin/bash
set -e -x

# =============================================================================
# SkipElmanSilu 1B Training - Skip connection for gradient flow!
# =============================================================================
# Model: ~1B params using haste SkipElman CUDA kernels
# Architecture:
#   - Recurrence: h = z*h + (1-z)*a (SKIP CONNECTION for gradients!)
#   - Output: h * silu(gate) (input-dependent selection)
# Simpler than GRU (no reset gate), but has gradient highway like GRU
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_skip_elman_silu_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29530

echo "======================================================================="
echo "=== SkipElmanSilu 1B Training (skip connection for gradient flow!) ==="
echo "======================================================================="

# SkipElmanSilu 1B config: dim=2048, depth=32 gives ~1.01B params
# Harmonized with Mamba2: lr=0.0006, weight_decay=0.1, no TBPTT

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
  --use_skip_elman_silu \
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
  2>&1 | tee "logs/skip_elman_silu_1b_${TIMESTAMP}.log"

echo "Done!"

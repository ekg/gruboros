#!/bin/bash
set -e -x

# =============================================================================
# ElmanSilu 1B Training - Harmonized with Mamba2 baseline
# =============================================================================
# Model: ~1B params using haste ElmanSilu CUDA kernels
# Architecture: h_new = tanh(h_candidate) * silu(gate) - simpler than GRU
# Note: TBPTT disabled for fair comparison with Mamba2 (which can't do TBPTT)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_silu_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29522

echo "======================================================================="
echo "=== ElmanSilu 1B Training (haste CUDA kernels, 3x faster than GRU) ==="
echo "======================================================================="

# ElmanSilu 1B config: dim=2048, depth=32 gives ~1.01B params
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
  --use_elman_silu \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 48 \
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
  2>&1 | tee "logs/elman_silu_1b_${TIMESTAMP}.log"

echo "Done!"

#!/bin/bash
set -e -x

# =============================================================================
# LOG-SPACE GRU TEST - Small scale test with Pile data
# =============================================================================
#
# Quick test to verify log-space training works with real data
# Uses tiktoken tokenizer and Pile dataset (same as production)
#
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_logspace_test"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
NUM_GPUS=1

echo "======================================================================="
echo "=== LOG-SPACE GRU TEST ==="
echo "======================================================================="
echo "Architecture: 8 layers x (LayerNorm -> LogSpaceGRU -> Residual)"
echo "Dim: 512, Expansion: 1.5"
echo "Testing with real data: Pile + tiktoken"
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
  --dim 512 \
  --depth 8 \
  --expansion_factor 1.5 \
  --dropout 0.0 \
  \
  --use_logspace_gru \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 1000 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 5 \
  --keep_elite 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "Log-space test complete!"

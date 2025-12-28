#!/bin/bash
set -e -x

# =============================================================================
# ELMAN LADDER LEVEL 1-LOG: Gated Elman + Log-Space Storage
# =============================================================================
# Delta gate + log-space hidden state storage
# Tests whether log-space storage helps with gated recurrence
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_ladder_level1_log"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== LADDER LEVEL 1-LOG: Gated Elman + Log-Space Storage ==="
echo "======================================================================="
echo "Architecture: delta gate + log-space hidden state"
echo "Dim: 1536, Depth: 20, Expansion: 1.5"
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
  --dim 1536 \
  --depth 20 \
  --expansion_factor 1.5 \
  --dropout 0.1 \
  \
  --elman_ladder_level 1-log \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 32 \
  --grad_accum 4 \
  \
  --train_steps 10000 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --keep_elite 5 \
  --milestone_every 2000 \
  \
  --ddp \
  --cuda \
  --bf16

echo "Level 1-log (Gated Elman + Log-Space) training complete!"

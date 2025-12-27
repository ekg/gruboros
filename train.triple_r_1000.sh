#!/bin/bash
set -e -x

# =============================================================================
# Elman Triple R - Extended 1000 step run
# =============================================================================
# Comparing with Selective Triple R at 1000 steps
# This is the original Triple R without the Mamba2-style B gate
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_triple_r_1000"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "======================================================================="
echo "=== Elman Triple R Extended (1000 steps) ==="
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
  --depth 34 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_triple_r \
  --compete_n_groups 32 \
  --delta_init -1.8 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 1000 \
  --lr 0.0006 \
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
  2>&1 | tee "logs/triple_r_1000_${TIMESTAMP}.log"

echo "Done!"

#!/bin/bash
set -e -x

# =============================================================================
# Selective Triple R ~1.33B params - Mamba2-style input selectivity + Triple R
# =============================================================================
# depth=30 gives 1.339B params (matched to Mamba2 1.33B)
# Triple R architecture + input-dependent B gate (like Mamba2's B projection)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_selective_triple_r_1.33b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "======================================================================="
echo "=== Selective Triple R (depth=30, ~1.33B params) ==="
echo "======================================================================="
echo "Triple R + input-dependent B gate (Mamba2-style selectivity)"
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
  --depth 30 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_selective_triple_r \
  --compete_n_groups 32 \
  --delta_init -1.8 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 300 \
  --lr 0.0006 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 100 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/selective_triple_r_1.33b_${TIMESTAMP}.log"

echo "Done!"

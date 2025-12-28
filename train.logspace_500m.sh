#!/bin/bash
set -e -x

# =============================================================================
# 500M LOG-SPACE TRAINING - Realistic scale test
# =============================================================================
#
# Tests log-space GRU with LogRMSNorm at 500M parameter scale
# Compare with train.baseline_500m.sh for fair comparison
#
# Config: dim=1536, depth=20, expansion=1.5 → ~500M params
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_logspace_500m"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== 500M LOG-SPACE TRAINING ==="
echo "======================================================================="
echo "Architecture: 20 layers × (LayerNorm → LogSpaceGRU+LogRMSNorm → Residual)"
echo "Dim: 1536, Expansion: 1.5, Inner: 2304"
echo "Parameters: ~500M"
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
  --use_logspace_gru \
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
  --ddp-find-unused \
  --cuda \
  --bf16

echo "500M log-space training complete!"

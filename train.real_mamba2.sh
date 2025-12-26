#!/bin/bash
set -e -x

# =============================================================================
# REAL Mamba2 (from mamba-ssm) - ~1.3B params to match Triple R
# =============================================================================
# Uses actual Mamba2 with SSD (State Space Duality)
# depth=48 gives 1.33B params vs Triple R's 1.28B
# Same dim, batch_size, chunk_size as Triple R for fair comparison
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_real_mamba2"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29705

echo "======================================================================="
echo "=== REAL Mamba2 (dim=2048, depth=48, 1.33B params) ==="
echo "======================================================================="
echo "Using official mamba-ssm Mamba2 implementation"
echo "d_state=64, expand=2, headdim=64 (Mamba2 defaults)"
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
  --depth 48 \
  --mamba_d_state 64 \
  --mamba_expand 2 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_mamba2 \
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
  2>&1 | tee "logs/real_mamba2_${TIMESTAMP}.log"

echo "Done!"

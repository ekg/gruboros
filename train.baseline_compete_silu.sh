#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeakyCompeteSilu Baseline - ~1.15B params
# =============================================================================
# Baseline for comparison with Triple R, Mamba2, LLaMA
# Same config as Triple R but single R matrix (not triple)
# depth=32, dim=2048, expansion_factor=1.0, compete_n_groups=32
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_baseline_compete_silu"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

echo "======================================================================="
echo "=== ElmanLeakyCompeteSilu Baseline (dim=2048, depth=32, ~1.15B) ==="
echo "======================================================================="
echo "Single R matrix + compete×silu gate"
echo "For comparison with Triple R (3 R matrices)"
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
  --depth 32 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_leaky_compete_silu \
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
  2>&1 | tee "logs/baseline_compete_silu_${TIMESTAMP}.log"

echo "Done!"

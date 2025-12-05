#!/bin/bash
set -e -x

# =============================================================================
# PyTorch cuDNN GRU + Selective SSM: CUMULATIVE Experiment (SSM replaces EMA)
# =============================================================================
# Replacing EMA track with Selective Diagonal SSM:
# - GRU path: cuDNN GRU (nonlinear, sequential)
# - SSM path: Selective diagonal SSM (learned decay, input-dependent like Mamba)
#
# SSM advantages over EMA:
# - Learned decay rates (not fixed alpha=0.01)
# - Input-dependent "selectivity" (decay varies per-token)
# - Output projection for capacity
#
# Config: ff_mult=4, depth=16 for ~1.41B params (vs 1.28B for EMA)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_ssm_ff4"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512

echo "======================================================================="
echo "=== PyTorch cuDNN GRU+SSM 1.41B + FF_MULT=4 Training ==="
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
  --depth 16 \
  --expansion_factor 1.0 \
  --ff_mult 4.0 \
  --dropout 0.0 \
  \
  --use_cudnn_ssm_gru \
  \
  --chunk_size 512 \
  --batch_size 24 \
  --grad_accum 1 \
  \
  --train_steps 10000 \
  --lr 0.001 \
  --weight_decay 0.033 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/cudnn_ssm_ff4_${TIMESTAMP}.log"

echo "Done!"

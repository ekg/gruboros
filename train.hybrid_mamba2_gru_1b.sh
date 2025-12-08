#!/bin/bash
set -e -x

# =============================================================================
# Hybrid Mamba2 + cuDNN GRU Training - DDP mode with 8 GPUs
# =============================================================================
# Model: ~1.1B params with parallel Mamba2 and cuDNN GRU paths
# Each layer has both paths running in parallel with learned mixing weights
# Mamba2: Linear SSM for efficient long-range context
# cuDNN GRU: Nonlinear gating for complex pattern learning
# The model learns when to use which path during training
# Note: No TBPTT - processes full chunk_size as context (like Mamba2)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_hybrid_mamba2_gru_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29522

echo "======================================================================="
echo "=== Hybrid Mamba2 + cuDNN GRU 1.1B Training ==="
echo "======================================================================="

# Hybrid 1.1B config: dim=2048, depth=20
# depth=20 gives 1.12B params (Mamba2: 512M + GRU: 504M)
# Matches Mamba2 structure: no FFN, same d_state=64, expand=2

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
  --depth 20 \
  --dropout 0.0 \
  \
  --use_hybrid_mamba2_gru \
  --mamba_d_state 64 \
  --mamba_expand 2 \
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
  2>&1 | tee "logs/hybrid_mamba2_gru_1b_${TIMESTAMP}.log"

echo "Done!"

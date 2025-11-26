#!/bin/bash
set -e -x

# =============================================================================
# SMALL LOG-SPACE TEST - Quick Gradient Stability Validation
# =============================================================================
#
# Quick test to verify:
# 1. Log-space GRU works correctly
# 2. Residuals + LayerNorm integrate properly
# 3. Gradients stay healthy over 1000 steps
# 4. No NaN/Inf issues
#
# Configuration: 3 layers, dim=512, ~36M params
# Expected time: ~10-15 minutes on 8 GPUs
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_logspace_test_small"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== SMALL LOG-SPACE TEST ==="
echo "======================================================================="
echo "Architecture: 3 layers × (LayerNorm → LogSpaceGRU → Residual)"
echo "Dim: 512, Expansion: 1.5, Dropout: 0.1"
echo "Parameters: ~36M"
echo "Goal: Verify gradients stay >0.1 for 1000 steps"
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  `# SMALL TEST MODEL` \
  --dim 512 \
  --depth 3 \
  --expansion_factor 1.5 \
  --dropout 0.1 \
  \
  `# USE LOG-SPACE GRU` \
  --use_logspace_gru \
  \
  `# SEQUENCES - maximize throughput` \
  --chunk_size 512 \
  --batch_size 128 \
  --grad_accum 8 \
  \
  `# TRAINING - SHORT TEST` \
  --train_steps 1000 \
  --lr 0.001 \
  --weight_decay 0.033 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING - frequent for monitoring` \
  --save_every 100 \
  --keep_checkpoints 10 \
  \
  `# FLAGS` \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "✅ Small test complete! Check gradients in logs."

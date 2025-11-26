#!/bin/bash
set -e -x

# =============================================================================
# 700M LOG-SPACE TEST - Prove We Can Train Where We Failed Before
# =============================================================================
#
# SAME configuration as the failing run (20 layers, dim=2048)
# BUT with log-space + residuals + LayerNorm
#
# Original run: Stuck at loss 4.7-4.8, gradients died to 0.06-0.07
# Expected now: Loss improves, gradients stay >0.1
#
# Configuration: 20 layers, dim=2048, expansion=1.0, ~700M params
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_logspace_700m"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== 700M LOG-SPACE TEST - REDEMPTION RUN ==="
echo "======================================================================="
echo "Architecture: 20 layers × (LayerNorm → LogSpaceGRU → Residual)"
echo "Dim: 2048, Expansion: 1.0, Dropout: 0.0"
echo "Parameters: ~700M (SAME as failed run)"
echo ""
echo "Previous failure: Loss stuck at 4.7-4.8, gradients 0.06-0.07"
echo "Expected now: Loss decreases, gradients stay >0.1"
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
  `# SAME AS FAILED RUN` \
  --dim 2048 \
  --depth 20 \
  --expansion_factor 1.0 \
  --dropout 0.0 \
  \
  `# KEY DIFFERENCE: USE LOG-SPACE GRU` \
  --use_logspace_gru \
  \
  `# SEQUENCES - same as before` \
  --chunk_size 512 \
  --batch_size 114 \
  --grad_accum 16 \
  \
  `# TRAINING - same optimizer config` \
  --train_steps 1000000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.033 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING` \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --keep_elite 32 \
  --milestone_every 10000 \
  \
  `# FLAGS` \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "✅ 700M log-space training complete!"

#!/bin/bash
set -e -x

# =============================================================================
# 1B LOG-SPACE TRAINING - Full Scale Deep GRU
# =============================================================================
#
# Scaled up configuration with Mamba-inspired architecture
# 24 layers (vs 20), wider (dim=1920), higher expansion (1.5)
# Dropout added (0.2) for regularization at scale
#
# Batch size reduced to fit 1B model in memory
#
# Configuration: 24 layers, dim=1920, expansion=1.5, ~1.2B params
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_logspace_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== 1B LOG-SPACE TRAINING ==="
echo "======================================================================="
echo "Architecture: 24 layers × (LayerNorm → LogSpaceGRU → Residual)"
echo "Dim: 1920, Expansion: 1.5, Inner: 2880"
echo "Dropout: 0.2"
echo "Parameters: ~1.2B total (1.1B GRU + 96M embeddings)"
echo ""
echo "Batch size: 64 per GPU (reduced from 114 to fit memory)"
echo "Gradient accumulation: 16 (effective batch = 64×8×16 = 8192)"
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
  `# 1B MODEL CONFIGURATION` \
  --dim 1920 \
  --depth 24 \
  --expansion_factor 1.5 \
  --dropout 0.2 \
  \
  `# USE LOG-SPACE GRU` \
  --use_logspace_gru \
  \
  `# SEQUENCES - grad_accum=4 for smoother updates` \
  --chunk_size 512 \
  --batch_size 64 \
  --grad_accum 4 \
  \
  `# TRAINING - Conservative LR=0.01 + grad accumulation for stability` \
  --train_steps 1000000 \
  --lr 0.01 \
  --sgd \
  --momentum 0.0 \
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

echo "✅ 1B log-space training complete!"

#!/bin/bash
set -e

# STANDARD CONTEXT model: chunk_size=2048, batch_size=8
# Balanced: good context length + fast training + batch diversity
# Config: dim=1536, depth=12, expansion=2.0, NO top-level conv (~818M params)

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_2048ctx"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  STANDARD CONTEXT - chunk_size=2048, batch_size=8"
echo "================================================================"
echo "  Config: dim=1536, depth=12, expansion=2.0"
echo "  Total params: ~818M"
echo ""
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: 8 per GPU"
echo "  Effective batch: 8 × 8 = 64"
echo "  Context length: 2048 tokens (16× longer than baseline)"
echo ""
echo "  Benefits:"
echo "  - Standard context length for most documents"
echo "  - 2× better batch diversity vs 4096 context"
echo "  - Faster training speed (shorter sequences)"
echo "  - Memory: ~21.8 GB (batch=7 was 19.07 GB)"
echo ""
echo "  CHECKPOINT STRATEGY (background saves):"
echo "  - Rolling: save_every=500, keep_checkpoints=3"
echo "  - Milestones: milestone_every=2000"
echo "================================================================"

/home/erikg/micromamba/envs/mingru/bin/torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --dim 1536 \
  --depth 12 \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  `# ARCHITECTURE: NO top-level conv, GRU with expansion` \
  --expansion_factor 2.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 0 \
  --dropout 0.0 \
  `# GRU: CausalConvGRU (has internal conv for gating)` \
  --use_causal_conv_gru \
  `# STANDARD CONTEXT CONFIGURATION` \
  --chunk_size 2048 \
  --batch_size 8 \
  --grad_accum 1 \
  `# TRAINING (10K steps)` \
  --train_steps 10000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  `# CHECKPOINTING` \
  --save_every 500 \
  --keep_checkpoints 3 \
  --milestone_every 2000 \
  --keep_elite 0 \
  --validation_interval 2000 \
  --validation_batches 0 \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

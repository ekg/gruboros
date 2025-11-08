#!/bin/bash
set -e

# CLEAN 1.1B model: dim=2048, depth=12, NO expansion, NO gradient clipping
# Simple & stable: wider hidden state, no architectural complexity
# Config: dim=2048, depth=12, expansion=1.0, NO top-level conv (~1.1B params)

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_1b_clean"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  CLEAN 1.1B MODEL - dim=2048, batch_size=16"
echo "================================================================"
echo "  Config: dim=2048, depth=12, expansion=1.0"
echo "  Total params: ~1.1B"
echo ""
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: 16 per GPU"
echo "  Effective batch: 16 × 8 = 128"
echo "  Context length: 2048 tokens"
echo ""
echo "  TRAINING: 100,000 steps (~26.2B tokens, ~43 hours)"
echo ""
echo "  Architecture improvements:"
echo "  - NO expansion (simpler, faster)"
echo "  - Wider hidden state (2048 vs 1536)"
echo "  - NO gradient clipping (natural learning)"
echo "  - 2× larger batch size (16 vs 8)"
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
  --dim 2048 \
  --depth 12 \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  `# ARCHITECTURE: NO top-level conv, NO expansion, just wide GRU` \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 0 \
  --dropout 0.0 \
  `# GRU: CausalConvGRU (has internal conv for gating)` \
  --use_causal_conv_gru \
  `# CONTEXT CONFIGURATION` \
  --chunk_size 2048 \
  --batch_size 16 \
  --grad_accum 1 \
  `# TRAINING (100K steps)` \
  --train_steps 100000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --weight_decay 0.01 \
  `# NO GRADIENT CLIPPING - let optimizer work naturally` \
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


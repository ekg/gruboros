#!/bin/bash
set -e

# Small model: dim=1536, depth=12 (~620M params actual)
# Fast training with batch_size=96

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_small"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  SMALL MODEL - dim=1536, depth=12"
echo "================================================================"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: 96 per GPU"
echo "  Effective batch: 96 × 8 = 768"
echo ""
echo "  CHECKPOINT STRATEGY (background saves):"
echo "  - Rolling: save_every=1000, keep_checkpoints=3"
echo "  - Milestones: milestone_every=10000"
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
  `# ARCHITECTURE` \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  `# GRU: CausalConvGRU (memory efficient for backprop)` \
  --use_causal_conv_gru \
  `# THROUGHPUT OPTIMIZATION: Short sequences + large batch` \
  --chunk_size 128 \
  --batch_size 96 \
  --grad_accum 1 \
  `# TRAINING (100K steps)` \
  --train_steps 100000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  `# BACKGROUND CHECKPOINT SAVES` \
  --save_every 1000 \
  --keep_checkpoints 3 \
  --milestone_every 10000 \
  --keep_elite 0 \
  --validation_interval 10000 \
  --validation_batches 0 \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

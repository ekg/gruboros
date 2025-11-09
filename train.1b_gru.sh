#!/bin/bash
set -e

# StandardGRU 1B model: TRUE RECURRENT with gradient checkpointing
# cuDNN backend with gradient checkpointing for memory efficiency
# Full 2048 context + document boundary support + TBPTT
# Config: dim=2048, depth=12, expansion=1.0, StandardGRU (~813M params)

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_1b_gru"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  StandardGRU 1B MODEL - TRUE RECURRENT!"
echo "================================================================"
echo "  Config: dim=2048, depth=12, expansion=1.0"
echo "  Total params: ~813M"
echo ""
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: 32 per GPU"
echo "  Effective batch: 32 × 8 = 256"
echo "  Context length: 2048 tokens (REAL recurrence via TBPTT!)"
echo ""
echo "  TRAINING: 100,000 steps (~52.4B tokens)"
echo ""
echo "  Architecture: StandardGRU (cuDNN + gradient checkpointing)"
echo "  - TRUE hidden state: h[t] = f(h[t-1], x[t])"
echo "  - cuDNN backend for performance"
echo "  - Gradient checkpointing for memory efficiency"
echo "  - TBPTT with chunk_size=512 tokens"
echo "  - Document boundary support (hidden state reset)"
echo "  - NO expansion (simpler, faster)"
echo "  - NO gradient clipping (natural learning)"
echo ""
echo "  CHECKPOINT STRATEGY (background saves):"
echo "  - Rolling: save_every=500, keep_checkpoints=3"
echo "  - Milestones: milestone_every=2000"
echo "================================================================"

# Ensure conda environment is in PATH
export PATH="/home/erikg/micromamba/envs/mingru/bin:$PATH"

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
  `# ARCHITECTURE: StandardGRU with gradient checkpointing` \
  --expansion 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 0 \
  --dropout 0.0 \
  `# StandardGRU: cuDNN + gradient checkpointing (trades compute for memory)` \
  --use_standard_gru \
  --use_gradient_checkpointing \
  `# CONTEXT CONFIGURATION - Balance memory vs throughput` \
  --chunk_size 512 \
  --batch_size 32 \
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

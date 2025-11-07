#!/bin/bash
set -e -x

# --- 100K STEP PRODUCTION RUN ---
# Goal: Train 500M model for 100K steps (~20 hours)
# Rolling: save_every=1000, keep_checkpoints=10
# Milestone: milestone_every=10000 (permanent)
# Batch size: 96 (increased from 64, safe given memory usage)

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Create output directory with timestamp
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_bootstrap_100k"
mkdir -p "${OUTPUT_DIR}"

echo "Output directory: $OUTPUT_DIR"

# --- Distributed Settings ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export TORCH_DISTRIBUTED_TIMEOUT=3600s
export TORCH_DISTRIBUTED_BACKEND="gloo"
NUM_GPUS=8

# --- DISABLE torch.compile for now (faster startup) ---
export PYTORCH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 100K STEP BOOTSTRAP TRAINING ==="
echo "Configuration:"
echo "  - Model: 500M parameters (CausalConvGRU)"
echo "  - Steps: 100,000 (~20 hours)"
echo "  - Batch size: 96 (increased from 64)"
echo "  - Rolling: save_every=1000, keep_checkpoints=10"
echo "  - Milestone: milestone_every=10000 (permanent)"
echo "========================================"

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params 500m \
  \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  \
  `# ARCHITECTURE (SAME AS MEZO for compatibility)` \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  \
  `# GRU: CausalConvGRU (memory efficient for backprop)` \
  --use_causal_conv_gru \
  \
  `# THROUGHPUT OPTIMIZATION: Short sequences + large batch` \
  --chunk_size 128 \
  --batch_size 96 \
  --grad_accum 1 \
  \
  `# TRAINING (100K steps for overnight run)` \
  --train_steps 100000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  \
  `# TWO-TIER CHECKPOINT SYSTEM` \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --milestone_every 10000 \
  --keep_elite 5 \
  --validation_interval 10000 \
  --validation_batches 0 \
  \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Checking checkpoint files..."
ls -lh "$OUTPUT_DIR/checkpoints/" | grep -E "milestone_|latest|best"
echo ""
echo "Milestone checkpoints saved at steps: 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000"
echo "Rolling checkpoints: last 10 saves (every 1000 steps)"

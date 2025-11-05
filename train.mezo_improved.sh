#!/bin/bash
set -e -x

# --- MeZO 100K: FIXED EPSILON + HIGHER LR ---
# Previous run (epsilon=0.0001) plateaued at loss ~5.2 after 40K steps
# Research shows MeZO standard epsilon = 0.001 (we were 10× too small!)
#
# Changes from previous (failed) run:
# - zo_epsilon: 0.001 (10× increase - matches MeZO paper standard)
# - lr: 0.0002 (2× increase for faster learning)
# - grad_accum=32, batch_size=16 (keep proven smoothing config)

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Create output directory with timestamp
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mezo_100k"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/checkpoints"

echo "Output directory: $OUTPUT_DIR"

# --- Distributed Settings ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30408
export TORCH_DISTRIBUTED_TIMEOUT=3600s
export TORCH_DISTRIBUTED_BACKEND="gloo"
NUM_GPUS=8

# --- CRITICAL: DISABLE COMPILATION ---
export PYTORCH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== MeZO 100K PRODUCTION RUN ==="
echo "Architecture: 500M params, CausalConvGRU"
echo "Method: BATCHED parallel perturbations"
echo "Config: batch_size=16 × 8 GPUs = 128 perturbations/step"
echo "Grad accum: 32 → 4096 EFFECTIVE perturbations per update!"
echo "Learning rate: 0.0001 (proven stable)"
echo "torch.compile: DISABLED"
echo "Chunk size: 2048 tokens"
echo "Train steps: 100,000"
echo "Checkpoints: Every 1000 steps"
echo "Strategy: Proven config from 10K run (commit 0dca418)"
echo "Expected: loss ~2.0 or better at 100K steps"
echo "================================================"

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
  `# ARCHITECTURE` \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  \
  `# GRU CONFIGURATION` \
  --use_causal_conv_gru \
  \
  `# ZERO-ORDER OPTIMIZATION (BATCHED!) - FIXED EPSILON!` \
  --zero_order \
  --zo_method mezo \
  --zo_epsilon 0.001 \
  --zo_num_perturbations_mezo 32 \
  \
  `# PROVEN STABLE CONFIG (commit 0dca418)` \
  --chunk_size 2048 \
  --batch_size 16 \
  --grad_accum 32 \
  \
  `# TRAINING (100K steps, 10× epsilon + 2× LR)` \
  --train_steps 100000 \
  --lr 0.0002 \
  --sf_beta 0.9 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING` \
  --save_every 1000 \
  --keep_checkpoints 100 \
  --validation_interval 100000 \
  --validation_batches 0 \
  \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

echo ""
echo "=== 100K RUN COMPLETE ==="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints saved every 1000 steps"

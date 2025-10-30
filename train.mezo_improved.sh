#!/bin/bash
set -e -x

# --- MeZO AGGRESSIVE: Finding the Explosion Boundary ---
# Strategy: Push the limits!
# - grad_accum=32 (MASSIVE smoothing, 2× better than baseline)
# - lr=0.0003 (3× faster than stable 0.0001!)
# - batch_size=16 (unchanged, 128 total perturbations)
# Total effective perturbations: 32 × 8 GPUs × 16 = 4096!
# Goal: Find explosion boundary, maximize throughput!

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Create output directory with timestamp
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mezo_aggressive"
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

echo "=== MeZO AGGRESSIVE RUN ==="
echo "Architecture: 500M params, CausalConvGRU"
echo "Method: BATCHED parallel perturbations"
echo "Config: batch_size=16 × 8 GPUs = 128 perturbations/step"
echo "Grad accum: 32 → 4096 EFFECTIVE perturbations per update!"
echo "Learning rate: 0.0003 (3× FASTER than stable!)"
echo "torch.compile: DISABLED"
echo "Chunk size: 2048 tokens"
echo "Train steps: 10,000"
echo "Checkpoints: Every 500 steps"
echo "Strategy: Find explosion boundary, maximize throughput!"
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
  `# ZERO-ORDER OPTIMIZATION (BATCHED!)` \
  --zero_order \
  --zo_method mezo \
  --zo_epsilon 0.0001 \
  --zo_num_perturbations_mezo 32 \
  \
  `# ULTRA-SMOOTH CONFIG (Option C)` \
  --chunk_size 2048 \
  --batch_size 16 \
  --grad_accum 32 \
  \
  `# TRAINING (10K steps, ULTRA-LOW LR!)` \
  --train_steps 10000 \
  --lr 0.0003 \
  --sf_beta 0.9 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING` \
  --save_every 500 \
  --keep_checkpoints 20 \
  --validation_interval 100000 \
  --validation_batches 0 \
  \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

echo ""
echo "=== AGGRESSIVE RUN COMPLETE ==="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints saved every 500 steps"

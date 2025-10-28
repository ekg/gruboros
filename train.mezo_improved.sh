#!/bin/bash
set -e -x

# --- MeZO STABLE: Conservative LR + Strong Smoothing ---
# Strategy: Avoid instability from previous lr=0.001 explosion
# - grad_accum=16 (4× more smoothing, better gradient estimates)
# - lr=0.0001 (10× lower than unstable 0.001, same as baseline)
# - batch_size=16 (unchanged, 128 total perturbations)
# Total effective perturbations: 16 × 8 GPUs × 16 = 2048!

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Create output directory with timestamp
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mezo_stable"
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

echo "=== MeZO STABLE RUN ==="
echo "Architecture: 500M params, CausalConvGRU"
echo "Method: BATCHED parallel perturbations"
echo "Config: batch_size=16 × 8 GPUs = 128 perturbations/step"
echo "Grad accum: 16 → 2048 EFFECTIVE perturbations per update"
echo "Learning rate: 0.0001 (STABLE - avoid lr=0.001 explosion!)"
echo "torch.compile: DISABLED"
echo "Chunk size: 2048 tokens"
echo "Train steps: 10,000"
echo "Checkpoints: Every 500 steps"
echo "========================================="

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
  `# IMPROVED CONFIG` \
  --chunk_size 2048 \
  --batch_size 16 \
  --grad_accum 16 \
  \
  `# TRAINING (10K steps, STABLE LR!)` \
  --train_steps 10000 \
  --lr 0.0001 \
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
echo "=== IMPROVED RUN COMPLETE ==="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints saved every 500 steps"

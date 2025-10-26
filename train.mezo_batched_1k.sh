#!/bin/bash
set -e -x

# --- MeZO BATCHED 1000-STEP TEST ---
# Production-ready long run with deterministic checkpointing
# Expected time: 1000 steps × 6 sec/step = ~100 minutes (1.7 hours)

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Create output directory with timestamp
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mezo_batched_1k"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/checkpoints"

echo "Output directory: $OUTPUT_DIR"

# --- Distributed Settings ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30403
export TORCH_DISTRIBUTED_TIMEOUT=3600s
export TORCH_DISTRIBUTED_BACKEND="gloo"
NUM_GPUS=8

# --- Memory optimizations ---
export PYTORCH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== MeZO BATCHED 1000-STEP PRODUCTION RUN ==="
echo "Architecture: 500M params, CausalConvGRU"
echo "Method: BATCHED parallel perturbations"
echo "Config: batch_size=8 → 64 total perturbations (8×8 GPUs)"
echo "Chunk size: 2048 tokens"
echo "Checkpointing: Every 1000 steps (deterministic)"
echo "Expected time: ~100 minutes"
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
  `# BATCHED PERTURBATIONS` \
  --chunk_size 2048 \
  --batch_size 8 \
  --grad_accum 1 \
  \
  `# TRAINING (1000 steps!)` \
  --train_steps 1000 \
  --lr 0.0001 \
  --sf_beta 0.9 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING (every 1k steps, keep all)` \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --validation_interval 10000 \
  --validation_batches 32 \
  \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

echo ""
echo "=== 1000-STEP RUN COMPLETE ==="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints saved at: ${OUTPUT_DIR}/checkpoints/"
ls -lh "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || echo "No checkpoints found"
echo ""
echo "Final steps:"
tail -10 "${OUTPUT_DIR}/metrics/training_metrics_rank_000.tsv" 2>/dev/null || echo "No metrics file found"

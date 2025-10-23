#!/bin/bash
set -e -x

# --- MeZO (Memory-Efficient Zeroth-Order) Training Script ---
# Based on NeurIPS 2023 paper: "Fine-Tuning Language Models with Just Forward Passes"
#
# Key features:
# - Same memory as inference (no gradients, no backward pass!)
# - Only 2 forward passes per step
# - Scales linearly to thousands of GPUs (no gradient sync)
# - Perfect for unbounded context training
#
# Test configuration: 500M parameters, 8 GPUs, 100 steps

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="500m"
NAME="${PARAMS}_mezo"

# Try to get git commit hash (first 7 chars)
GIT_HASH=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")
fi

# Build output directory name with optional git hash
if [ -n "$GIT_HASH" ]; then
    OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_${NAME}"
fi
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

### Explicitly define and manage a temp directory ###
# Create a unique, job-specific temporary directory in /tmp
JOB_ID=$(date +%s) # Simple job ID using timestamp for local runs
GOSSIP_TEMP_DIR="/tmp/gossip_temp_${JOB_ID}"
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${GOSSIP_TEMP_DIR}"
echo "Using local temporary directory: $GOSSIP_TEMP_DIR"

# --- Distributed Settings for Launcher & Script ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_TIMEOUT=3600s
# Use GLOO for peer discovery.
export TORCH_DISTRIBUTED_BACKEND="gloo"
echo "Using GLOO backend for initial process group."
NUM_GPUS=8

# --- torch.compile Caching ---
# Cache compiled kernels to speed up subsequent runs
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_DIR}/.inductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
echo "torch.compile cache: $TORCHINDUCTOR_CACHE_DIR"

# --- Launch Training with MeZO ---
echo "Starting 500M parameter MeZO test on 8 GPUs."
echo "Architecture: StandardGRU, NO FFN, TikToken (100K vocab)"
echo "Configuration: batch_size=4, chunk_size=2048 (same as standard backprop for comparison)"
echo "Method: MeZO in-place perturbation (2 forward passes per step)"
echo "Memory: Same as inference! Unbounded context possible!"
echo "Test duration: 100 steps"

torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params $PARAMS \
  \
  `# TOKENIZATION (CRITICAL - enables semantic learning)` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  \
  `# ARCHITECTURE (NO FFN!)` \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  \
  `# GRU CONFIGURATION (standard GRU with cuDNN)` \
  --use_standard_gru \
  \
  `# ZERO-ORDER OPTIMIZATION (MeZO!)` \
  --zero_order \
  --zo_method mezo \
  --zo_epsilon 0.0001 \
  \
  `# SEQUENCES (same as standard backprop for fair comparison)` \
  --chunk_size 2048 \
  --batch_size 4 \
  --grad_accum 1 \
  \
  `# TRAINING (test multi-perturbation)` \
  --train_steps 20 \
  --lr 0.0001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING` \
  --save_every 1000 \
  --keep_checkpoints 5 \
  --keep_elite 32 \
  --archive_rate 0.0067 \
  --validation_interval 500 \
  --validation_batches 32 \
  \
  `# GOSSIP (evolutionary training)` \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.2 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.0003 \
  --gossip_p_value_threshold 0.1 \
  --gossip_lock_timeout 5.0 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --gossip_fitness_window 10000 \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 20.0 \
  \
  `# FLAGS` \
  --ddp \
  --cuda

echo "Training finished."
echo "Expected performance:"
echo "  - Memory: ~10 GB (same as inference!)"
echo "  - Time per step: ~4-5s (2× forward passes)"
echo "  - Comparison: Standard backprop ~2s, but requires gradients"
echo "  - MeZO advantage: Can train with UNBOUNDED context!"

### Clean up the temporary directory ###
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up local temporary directory."

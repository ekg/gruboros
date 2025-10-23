#!/bin/bash
set -e -x

# --- MeZO (Memory-Efficient Zeroth-Order) Training ---
# Based on train.standard_gru.sh but using zero-order optimization
# Forward-only training with same memory as inference!

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

### Create output directories ###
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/gossip"

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

# --- Launch MeZO Training ---
echo "Starting 500M parameter MeZO training on 8 GPUs."
echo "Architecture: StandardGRU, NO FFN, TikToken (100K vocab)"
echo "Method: Zero-order optimization (forward-only, same memory as inference!)"

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
  `# SEQUENCES (multi-perturbation via grad_accum)` \
  --chunk_size 2048 \
  --batch_size 2 \
  --grad_accum 4 \
  \
  `# TRAINING` \
  --train_steps 100000 \
  --lr 0.0001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING` \
  --save_every 500 \
  --keep_checkpoints 5 \
  --validation_interval 5000 \
  --validation_batches 32 \
  \
  `# FLAGS` \
  --ddp \
  --cuda

echo "Training finished."
echo "MeZO Training Complete!"
echo "  - Memory: Same as inference + momentum buffers (~5GB extra for Adam)"
echo "  - Forward passes per step: $(( 2 * 4 )) (4 perturbations)"
echo "  - Data throughput: $(( 2 * 8 * 2048 * 4 )) tokens/step (4 fresh batches)"
echo "  - Compute: $(( 2 * 8 * 2048 * 8 )) tokens of forward passes per step"

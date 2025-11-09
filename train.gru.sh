#!/bin/bash
set -e -x

# --- Optimized HybridGRU Training Script ---
# Configuration:
# - p50k_base tokenizer (50,257 tokens) - 50% smaller vocab, faster softmax
# - depth=20 (vs 12) - 67% deeper model for better representations
# - dim=2048, ~709M params
# - chunk_size=512 - 2× more gradient updates per epoch
# - z_bias=0.0 - let model learn optimal update/forget balance
# - DDP pipeline optimizations (static_graph, prefetch, persistent workers)

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="700m"
NAME="${PARAMS}_gru_deep"

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

# --- Launch Training with HybridGRU Architecture ---
echo "Starting ~709M parameter HybridGRU training on 8 GPUs."
echo "Architecture: depth=20, dim=2048, p50k_base tokenizer, 512 token chunks"
echo "Optimizations: z_bias=0.0, DDP pipeline (static_graph, prefetch)"

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  `# TOKENIZATION (p50k_base - 50% smaller vocab)` \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  `# ARCHITECTURE (Deep HybridGRU)` \
  --dim 2048 \
  --depth 20 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  `# GRU CONFIGURATION (z_bias=0.0 for adaptive gating)` \
  --z_bias_input 0.0 \
  --z_bias_hidden 0.0 \
  --hybrid_gru \
  \
  `# SEQUENCES (512 tokens, 1M tokens per update)` \
  --chunk_size 512 \
  --batch_size 128 \
  --grad_accum 16 \
  \
  `# TRAINING` \
  --train_steps 100000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.033 \
  --grad_clip 1.0 \
  \
  `# DDP OPTIMIZATIONS` \
  --num_workers 4 \
  \
  `# CHECKPOINTING` \
  --save_every 500 \
  --keep_checkpoints 5 \
  --keep_elite 32 \
  --archive_rate 0.0067 \
  --validation_interval 2000 \
  --validation_batches 32 \
  \
  `# GOSSIP (evolutionary training - DISABLED for clean baseline)` \
  --gossip_mixing_rate 0.0 \
  \
  `# FLAGS` \
  --ddp \
  --ddp-find-unused \
  --cuda

echo "Training finished."

### Clean up the temporary directory ###
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up local temporary directory."

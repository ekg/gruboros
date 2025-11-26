#!/bin/bash
set -e -x

# =============================================================================
# Train 1B Log-Space Deep GRU with Mamba-style Architecture
# =============================================================================
#
# Key innovations:
# - Log-space hidden states (no vanishing gradients!)
# - Residual connections between layers (Mamba-style)
# - Layer normalization (pre-norm)
# - 24 layers deep (enabled by above features)
#
# Expected: Gradient norms stay healthy (>0.1) throughout training
#           Loss decreases consistently (no plateau at 4.7!)
#
# =============================================================================

# --- CHECKPOINT TO RESUME FROM (optional) ---
# RESUME_CHECKPOINT="/mnt/nvme2n1/erikg/minlms/PREVIOUS_RUN/latest.pt"
RESUME_CHECKPOINT=""

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="1b"
NAME="${PARAMS}_logspace_deep"

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
JOB_ID=$(date +%s)
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
export TORCH_DISTRIBUTED_BACKEND="nccl"
echo "Using NCCL backend for GPU communication."
NUM_GPUS=8

# --- torch.compile Caching ---
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_DIR}/.inductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
echo "torch.compile cache: $TORCHINDUCTOR_CACHE_DIR"

# --- Launch Training ---
echo "======================================================================="
echo "=== Training 1B Log-Space Deep GRU ==="
echo "======================================================================="
echo "Architecture: 24 layers × (LayerNorm → LogSpaceGRU → Residual)"
echo "Parameters: ~1.2B total (1.1B GRU + 96M embeddings)"
echo "Dim: 1920, Expansion: 1.5, Inner: 2880"
echo "Optimizer: Schedule-Free AdamW"
echo "Learning rate: 0.001"
echo "Gradient clipping: DISABLED (log-space handles stability)"
echo "Dropout: 0.2 (per minGRU paper)"
echo ""
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
else
    echo "Starting fresh training"
fi
echo "======================================================================="

RESUME_ARG=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ARG="--resume $RESUME_CHECKPOINT"
fi

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_logspace.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  `# ARCHITECTURE - 1B Log-Space Deep GRU` \
  --dim 1920 \
  --depth 24 \
  --expansion_factor 1.5 \
  --dropout 0.2 \
  \
  `# SEQUENCES` \
  --chunk_size 512 \
  --batch_size 64 \
  --grad_accum 16 \
  \
  `# TRAINING - Schedule-Free AdamW, NO GRAD CLIP` \
  --train_steps 1000000 \
  $RESUME_ARG \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.033 \
  --grad_clip 0.0 \
  \
  `# DDP` \
  --num_workers 4 \
  \
  `# CHECKPOINTING` \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --keep_elite 32 \
  --milestone_every 10000 \
  --archive_rate 0.0 \
  --validation_interval 1000000 \
  --validation_batches 8 \
  \
  `# GOSSIP DISABLED` \
  --gossip_mixing_rate 0.0 \
  \
  `# FLAGS` \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "Training finished."

### Clean up ###
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up local temporary directory."

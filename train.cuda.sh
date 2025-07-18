#!/bin/bash
set -e -x

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_ok"

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
# --- *** FIX: PRE-CREATE ALL DIRECTORIES *** ---
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

# --- Launch Training ---
echo "Starting 'Thin & Deep' run for emergent behaviors."
deepspeed --num_gpus=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params 1g \
  --dim 1024 \
  --expansion_factor 3.0 \
  --ff_mult 1.5 \
  --train_steps 2000000 \
  --save_every 500 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 8 \
  --chunk_size 2048 \
  --context_chunks 8 \
  --keep_checkpoints 5 \
  --keep_elite 10 \
  --archive_rate 0.02 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.3 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.002 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --gossip_fitness_decay 0.995 \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 5.0 \
  --cuda

echo "Training finished."

### Clean up the temporary directory ###
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up local temporary directory."

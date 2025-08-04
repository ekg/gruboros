#!/bin/bash
set -e -x

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="1g"
NAME="${PARAMS}"

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
echo "Starting 1B parameter pure RNN run with document-aware validation."

# Training configuration with exposed parameters for easy tweaking
deepspeed --num_gpus=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params $PARAMS \
  --dim 1536 \
  --expansion_factor 4.0 \
  --ff_mult 0 \
  --train_steps 10000000 \
  --save_every 500 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --grad_accum 1024 \
  --chunk_size 2048 \
  --keep_checkpoints 5 \
  --keep_elite 32 \
  --archive_rate 0.0067 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.2 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.002 \
  --gossip_p_value_threshold 0.2 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --gossip_fitness_window 10000 \
  --validation_sequences 32 \
  --validation_sequence_length 8k \
  --validation_interval 10000 \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 20.0 \
  --cuda

echo "Training finished."

### Clean up the temporary directory ###
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up local temporary directory."

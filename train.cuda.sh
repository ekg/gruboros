#!/bin/bash
set -e -x

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_8gpu_pure_gossip"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
mkdir -p logs "$OUTPUT_DIR"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

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
echo "Starting Pure Gossip training on 8 GPUs. DeepSpeed is used ONLY as a launcher."
deepspeed --num_gpus=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --train_steps 100k \
  --validate_every 1000 \
  --save_every 2000 \
  --lr 0.0001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 1 \
  --seq_len 2k \
  --params 1g \
  --keep_checkpoints 3 \
  --cuda

echo "Training finished."

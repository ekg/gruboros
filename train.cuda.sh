#!/bin/bash

# Enable error handling
set -e
set -x

# ====================== ENVIRONMENT SETUP ======================

# Create log directory
mkdir -p logs

# Generate timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_model"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
echo "Generated Output Directory: ${OUTPUT_DIR}"
mkdir -p ./outputs

# ====================== CUDA & DISTRIBUTED CONFIGURATION ======================

# NVIDIA/CUDA specific settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Force using gloo backend instead of NCCL
# Set to OFF to disable verbose comm op logs
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_DISTRIBUTED_DETAIL=0          # Disable detailed distributed logging
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions

# Explicitly disable NCCL
export NCCL_P2P_DISABLE=1
export USE_NCCL=0

# Set the PyTorch distributed backend to gloo
export TORCH_DISTRIBUTED_BACKEND="gloo"

# Prevent timeout issues during initialization
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600

# Performance tuning
export OMP_NUM_THREADS=4                  # Limit OpenMP threads to avoid CPU oversubscription

# ====================== DISTRIBUTED TRAINING SETUP ======================

# Set up master address and port
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Number of GPUs
NUM_GPUS=8

# ====================== TRAINING LAUNCH ======================

# Data path - MODIFY THIS TO YOUR DATA PATH
DATA_PATH="/mnt/nvme2n1/erikg/fineweb-edu/fineweb-0kx10m.txt"

echo "Starting training with DeepSpeed..."
deepspeed --num_gpus=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --cuda \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --train_steps 100000 \
  --validate_every 1000 \
  --save_every 2000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 32 \
  --seq_len 2k \
  --depth 8 \
  --params 2g \
  --tp_size $NUM_GPUS \
  --keep_checkpoints 5 \
  --deepspeed \
  --deepspeed_config ds_config.json

echo "Training finished."

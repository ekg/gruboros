#!/bin/bash

# Enable error handling
set -e
set -x

# ====================== ENVIRONMENT SETUP ======================

# Create log directory
mkdir -p logs

# Generate timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="512m_model"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
echo "Generated Output Directory: ${OUTPUT_DIR}"
mkdir -p ./outputs

# ====================== CUDA & NCCL CONFIGURATION ======================

# NVIDIA/CUDA specific settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings optimized for single-node multi-GPU setup
export NCCL_DEBUG=INFO                     # Set to INFO for diagnosing issues, WARN for production
export NCCL_SOCKET_IFNAME=lo               # Use loopback for single-node communication
export NCCL_IB_DISABLE=1                   # Disable InfiniBand since we're on a single node
export NCCL_P2P_DISABLE=0                  # Enable P2P for GPU-to-GPU communication 
export NCCL_P2P_LEVEL=5                    # Maximum P2P level
export TORCH_DISTRIBUTED_DETAIL=1          # Detailed distributed logging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # Enable async error handling
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions

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
  --lr 0.0015 \
  --sf_beta 0.9 \
  --weight_decay 0.01 \
  --batch_size 4 \
  --grad_accum 128 \
  --seq_len 2k \
  --params 512m \
  --tp_size $NUM_GPUS \
  --keep_checkpoints 5 \
  --deepspeed \
  --deepspeed_config ds_config.json

echo "Training finished."

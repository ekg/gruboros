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

# Force using gloo backend instead of NCCL for local testing
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_DISTRIBUTED_DETAIL=0
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions

# Explicitly disable NCCL for local testing (gossip works better with gloo)
export NCCL_P2P_DISABLE=1
export USE_NCCL=0

# Set the PyTorch distributed backend to gloo
export TORCH_DISTRIBUTED_BACKEND="gloo"

# Prevent timeout issues during initialization
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600

# Performance tuning
export OMP_NUM_THREADS=4

# ====================== LOCAL GOSSIP PROTOCOL SETUP ======================

# Create hostfile for local 8-GPU testing (required for gossip discovery)
echo "Creating local hostfile for gossip protocol..."
cat > hostfile-job-local.txt << EOF
localhost slots=8
EOF

# Set environment variables for gossip discovery
export SLURM_JOB_ID="local"
export SLURM_JOB_NODELIST="localhost"

# Gossip protocol will use ports 29501-29508 for ranks 0-7
echo "Gossip protocol will use ports 29501-29508 for 8 GPU ranks"

# ====================== DISTRIBUTED TRAINING SETUP ======================

# Set up master address and port for DeepSpeed
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500  # DeepSpeed uses this, gossip uses 29501+

# Number of GPUs
NUM_GPUS=8

# ====================== TRAINING LAUNCH ======================

# Data path - MODIFY THIS TO YOUR DATA PATH
DATA_PATH="/mnt/nvme2n1/erikg/fineweb-edu/fineweb-0kx10m.txt"

# Verify data path exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    echo "Please update DATA_PATH in train.cuda.sh to point to your data file"
    exit 1
fi

echo "Starting 8-GPU training with evolutionary gossip protocol..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Hostfile created: hostfile-job-local.txt"
deepspeed --num_gpus=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --cuda \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --train_steps 10000 \
  --validate_every 500 \
  --save_every 1000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 16 \
  --seq_len 1024 \
  --depth 8 \
  --params 1g \
  --tp_size $NUM_GPUS \
  --keep_checkpoints 3 \
  --deepspeed \
  --deepspeed_config ds_config.json

echo "Training finished."

# ====================== CLEANUP ======================

# Clean up temporary files
rm -f hostfile-job-local.txt

echo "Cleaned up temporary files."

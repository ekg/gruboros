#!/bin/bash

# Enable error handling
set -e
set -x

# ====================== ENVIRONMENT SETUP ======================

# Create log directory
mkdir -p logs

# Generate timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="16m_model_8way_dp_gossip"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
echo "Generated Output Directory: ${OUTPUT_DIR}"
mkdir -p ./outputs

# ====================== CUDA & DISTRIBUTED CONFIGURATION ======================

# NVIDIA/CUDA specific settings - USE GPUs 0-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Force using gloo backend for data parallel + gossip
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_DISTRIBUTED_DETAIL=0
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions

# Explicitly disable NCCL for local testing
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

# Create hostfile for local 8-GPU data parallel testing
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
DATA_PATH="/home/erikg/enwik/enwik8"

# Verify data path exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    echo "Please update DATA_PATH in train.cuda.sh to point to your data file"
    exit 1
fi

echo "Starting 8-way DATA PARALLEL training with evolutionary gossip..."
echo "Model setup: 8 independent 16M parameter models (one per GPU)"
echo "Using GPUs: 0,1,2,3,4,5,6,7"
echo "Tensor Parallel Size: 1 (each GPU has complete model)"
echo "Data Parallel Size: 8 (8 separate models with different data)"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Hostfile created: hostfile-job-local.txt"
# Launch training with DeepSpeed - DATA PARALLEL CONFIGURATION
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
  --lr 0.003 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 16 \
  --grad_accum 4 \
  --seq_len 2048 \
  --params 16m \
  --tp_size 1 \
  --keep_checkpoints 3 \
  --deepspeed \
  --deepspeed_config ds_config.json

echo "Training finished."

# ====================== CLEANUP ======================

# Clean up temporary files
rm -f hostfile-job-local.txt

echo "Cleaned up temporary files."

# ====================== POST-TRAINING SUMMARY ======================

echo "=== Training Summary ==="
echo "Configuration: 8-way Data Parallel"
echo "Model size: 16M parameters PER GPU (8 independent models)"
echo "GPUs used: 0,1,2,3,4,5,6,7"
echo "Tensor parallel size: 1 (complete model per GPU)"
echo "Data parallel size: 8 (8 different models)"
echo "Each model gets different data samples"
echo "Evolutionary mixing between the 8 models via gossip"
echo "Sequence length: 2048"
echo "Batch size per GPU: 16"
echo "Gradient accumulation: 4"
echo "Global effective batch size: 512 (8 * 16 * 4)"
echo "Learning rate: 0.003"
echo "Training steps: 100,000"
echo "Output saved to: $OUTPUT_DIR"

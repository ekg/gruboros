#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_frontier
#SBATCH -o logs/minLM_frontier-%j.out
#SBATCH -e logs/minLM_frontier-%j.err
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 8                  # Number of nodes
#SBATCH --ntasks-per-node=8   # CRITICAL: 8 GPUs per node
#SBATCH --gpus-per-node=8     # Explicitly request 8 GPUs per node
#SBATCH -q debug

# Enable command echoing for better debugging
set -x

# Setup Python environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros

# Preload libraries needed on OLCF systems
export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

# Load necessary modules
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Export general settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=2

# Set up distributed environment using Slurm
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=3442
export TORCH_DISTRIBUTED_TIMEOUT=3600s    # 1 hr timeout for initialization

# NCCL/RCCL settings optimized for Frontier's Slingshot fabric
export UCX_TLS=rc,tcp,sm
export NCCL_DEBUG=INFO                     # Set to INFO to diagnose distribution issues
export RCCL_DEBUG=INFO                     # ROCm-specific debug info
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1         # Both error flags for redundancy
export NCCL_TIMEOUT=10000000               # Collective timeout in ms
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=hsn0             # Primary high-speed network interface
export NCCL_CROSS_NIC=1                    # Enable multi-rail
export NCCL_NET_GDR_LEVEL=3                # RDMA via OFI
export NCCL_MIN_NCHANNELS=32               # MI250X tuning
export NCCL_CUMEM_ENABLE=0                 # Disable CUDA unified memory

# Performance tuning
export FI_CXI_ATS=0
export NCCL_COLLNET_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=1M
export NCCL_P2P_PCI_RELAXED_ORDERING=1

# MIOPEN cache paths per node to avoid contention
export MIOPEN_USER_DB_PATH="/tmp/${USER:-user}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_USER_DB_PATH"

# === ELASTICITY SETTINGS ===
export TORCH_DISTRIBUTED_AUTO_RESTART=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=3600       # 1 hour timeout in seconds
export NCCL_TIMEOUT=3600000                 # 1 hour timeout in milliseconds
# === END ELASTICITY SETTINGS ===

# === SLINGSHOT OPTIMIZATION SETTINGS (NEW) ===
# Multi-rail Slingshot configuration
export UCX_NET_DEVICES=hsn0,hsn1,hsn2,hsn3
export UCX_TLS=rc,ud,sm,self
export UCX_RNDV_SCHEME=get_zcopy

# RCCL/NCCL multi-rail and buffer optimization
export NCCL_IB_HCA=hsn0,hsn1,hsn2,hsn3
export NCCL_CROSS_NIC=1
export NCCL_P2P_NET_CHUNKSIZE=8M
export NCCL_BUFFSIZE=16777216

# ROCm-specific optimizations for Frontier's AMD GPUs
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=8
export NCCL_NVLS_ENABLE=0
# === END SLINGSHOT OPTIMIZATION SETTINGS ===

# Tell DeepSpeed to use Slurm launcher
export DEEPSPEED_USE_SRUN=1

# Setup torchrun port for rendezvous
export TORCHRUN_PORT=29500

# Generate timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1b_tweak"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
echo "Generated Output Directory: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

# Calculate total ranks for debugging
ranks_per_node=8
ranks_total=$((ranks_per_node*SLURM_JOB_NUM_NODES))
echo "Total ranks: $ranks_total (expected to use $SLURM_JOB_NUM_NODES nodes with $ranks_per_node ranks per node)"

# Set data path
DATA="/lustre/orion/bif148/scratch/erikgarrison/fineweb-edu/sample/10BT.txt"
echo "Using data: $DATA"

# Create log directories
mkdir -p logs
mkdir -p ./outputs

# === SIMPLE FILE-BASED RENDEZVOUS SETUP ===
# Create rendezvous directory with generous permissions
export RENDEZVOUS_DIR="/lustre/orion/bif148/scratch/$USER/rendezvous_$SLURM_JOB_ID"
echo "Using rendezvous directory: $RENDEZVOUS_DIR"

# Only rank 0 creates the directory
if [ "$SLURM_PROCID" -eq 0 ]; then
    mkdir -p $RENDEZVOUS_DIR
    chmod 777 $RENDEZVOUS_DIR
    echo "Created rendezvous directory: $RENDEZVOUS_DIR"
fi

# Simple barrier to ensure directory exists before proceeding
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 true
echo "All nodes synchronized after directory creation"
# === END SIMPLE FILE-BASED RENDEZVOUS SETUP ===

# Set data path
DATA="/lustre/orion/bif148/scratch/erikgarrison/fineweb-edu/sample/10BT.txt"
echo "Using data: $DATA"

# Calculate total ranks for debugging
ranks_per_node=8
ranks_total=$((ranks_per_node*SLURM_JOB_NUM_NODES))
echo "Total ranks: $ranks_total (expected to use $SLURM_JOB_NUM_NODES nodes with $ranks_per_node ranks per node)"

# Print SLURM environment for debugging
env | grep SLURM

# Launch with torchrun using file-based rendezvous
echo "Starting training with torchrun elastic launcher using file-based rendezvous..."
srun torchrun \
  --nproc_per_node=8 \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_conf="file_init=${RENDEZVOUS_DIR}" \
  --rdzv_id=$SLURM_JOB_ID \
  --max_restarts=3 \
  --monitor_interval=5 \
  train.py \
  --data "$DATA" \
  --output "$OUTPUT_DIR" \
  --train_steps 100000 \
  --validate_every 256 \
  --save_every 256 \
  --lr 0.02 \
  --sf_beta 0.84 \
  --weight_decay 5e-4 \
  --batch_size 6 \
  --grad_accum 8 \
  --gradient_clipping 1.0 \
  --seq_len 2048 \
  --params 1g \
  --tp_size 8 \
  --keep_checkpoints 5 \
  --deepspeed \
  --deepspeed_config ds_config.json

echo "Training finished."

# Clean up the rendezvous directory after completion
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Cleaning up rendezvous directory"
    rm -rf $RENDEZVOUS_DIR
fi

echo "Training completed with elastic launcher"

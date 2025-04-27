#!/bin/bash
# Wrapper script to ensure proper environment for deepspeed on each Slurm task

# Load necessary modules for execution environment
module load PrgEnv-amd
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"

# Set up micromamba - needs proper initialization
micromamba activate gruboros

# Set ROCm-specific environment variables
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export ROCR_LOG_LEVEL=INFO
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_USER_DB_PATH}"

# Critical network settings for Frontier's Slingshot fabric
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export LD_LIBRARY_PATH=/opt/rocm-6.2.4/rccl/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export FI_LOG_LEVEL=info

# Get the master node from SLURM
NODE_HOSTNAMES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
MASTER_NODE=$(echo $NODE_HOSTNAMES | awk '{print $1}')
export MASTER_ADDR=$MASTER_NODE

echo "Using hostname $MASTER_ADDR as MASTER_ADDR"

# Set fixed port for reproducibility
export MASTER_PORT=${MASTER_PORT:-29500}

echo "MASTER_ADDR set to: $MASTER_ADDR"
echo "MASTER_PORT set to: $MASTER_PORT"

# Create MIOPEN cache directory if it doesn't exist
mkdir -p $MIOPEN_USER_DB_PATH

# Print diagnostic information
echo "======== Task Environment ========"
echo "Running on node: $(hostname) with SLURM_LOCALID: $SLURM_LOCALID"
echo "DeepSpeed path: $(which deepspeed)"
echo "Python path: $(which python)"
echo "ROCm version: $(rocm-smi --showdriverversion 2>/dev/null || echo 'not available')"
echo "=================================="

# Run DeepSpeed with proper command structure
# --hostfile is optional but recommended for multi-node runs
if [ -f "hostfile.txt" ]; then
  deepspeed --hostfile hostfile.txt --num_gpus $SLURM_GPUS_PER_NODE --num_nodes $SLURM_NNODES "$@"
else
  deepspeed --num_gpus $SLURM_GPUS_PER_NODE --num_nodes $SLURM_NNODES "$@"
fi

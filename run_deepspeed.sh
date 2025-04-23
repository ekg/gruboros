#!/bin/bash
# Wrapper script to ensure proper environment for deepspeed on each Slurm task

# Load necessary modules for execution environment
module load PrgEnv-amd
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Set up micromamba - needs proper initialization
source $(micromamba info --base)/etc/profile.d/micromamba.sh
micromamba activate gruboros

# Set ROCm-specific environment variables
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export HSA_TOOLS_LIB=1
export ROCR_LOG_LEVEL=INFO
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_USER_DB_PATH}"

# Network configuration for Frontier's Slingshot network
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export NCCL_DEBUG=INFO

# Create MIOPEN cache directory if it doesn't exist
mkdir -p $MIOPEN_USER_DB_PATH

# Print diagnostic information
echo "======== Task Environment ========"
echo "Running on node: $(hostname) with SLURM_LOCALID: $SLURM_LOCALID"
echo "DeepSpeed path: $(which deepspeed)"
echo "Python path: $(which python)"
echo "ROCm version: $(rocm-smi --showdriverversion 2>/dev/null || echo 'not available')"
echo "=================================="

# Run the actual command, passing all arguments through
deepspeed train.py "$@"

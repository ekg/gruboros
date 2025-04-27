#!/bin/bash
# Simple wrapper script to launch minLM training on Frontier
# Designed to run directly on compute nodes

# Bail out immediately on errors
set -e

# Initialize logging
echo "========= minLM Training on Frontier ========="
echo "Date: $(date)"
echo "===== Environment Setup ====="

# Environment preparation 
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
export ROCM_HOME=/opt/rocm-6.2.4

# Setup for ROCm compatibility
export PYTORCH_ROCM_ARCH=gfx90a
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# Critical environment variables for distributed training
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export FI_LOG_LEVEL=info
export MPICH_GPU_SUPPORT_ENABLED=1

# Make training script executable
chmod +x srun_train.sh

# Print SLURM environment for debugging
echo "===== SLURM Environment ====="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# Check if running as SLURM job or interactive - if interactive, submit job
if [ -z "$SLURM_JOB_ID" ]; then
  echo "Not running as a SLURM job. Submitting batch job..."
  sbatch -A BIF148 -N 2 -t 01:00:00 -p batch -q debug --exclusive $0
  exit 0
fi

# Launch the training script directly if this is a slurm job
./srun_train.sh

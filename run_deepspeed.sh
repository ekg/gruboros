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

# Create a temporary hook script and source it directly
HOOK_SCRIPT=$(mktemp)
"$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" > "$HOOK_SCRIPT"
source "$HOOK_SCRIPT"
rm -f "$HOOK_SCRIPT"

# Set up micromamba - with fallback options
echo "Activating micromamba environment 'gruboros'..."
micromamba activate gruboros || (echo "Failed direct activation, trying alternative method" && \
  eval "$("$MAMBA_EXE" shell init --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" && \
  micromamba activate gruboros)

# Ensure Python is in PATH
export PATH="/lustre/orion/bif148/scratch/erikgarrison/micromamba/envs/gruboros/bin:$PATH"

# Set ROCm/HIP environment variables to use instead of CUDA
export CUDA_HOME=/opt/rocm-6.2.4
export HIP_CLANG_PATH=/opt/rocm-6.2.4/llvm
export ROCM_HOME=/opt/rocm-6.2.4
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Fix for DeepSpeed CUDA compatibility check
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0  # Disable building ops until compatibility is fixed

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

# DeepSpeed settings for ROCm compatibility
export DS_BUILD_OPS=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

# Get the master node from SLURM
NODE_HOSTNAMES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
MASTER_NODE=$(echo $NODE_HOSTNAMES | awk '{print $1}')
export MASTER_ADDR=$MASTER_NODE

echo "Using hostname $MASTER_ADDR as MASTER_ADDR"

# Set fixed port for reproducibility - use specific port to avoid conflicts
export MASTER_PORT=${MASTER_PORT:-3442}

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

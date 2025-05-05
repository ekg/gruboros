#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_frontier
#SBATCH -o logs/minLM_frontier-%j.out
#SBATCH -e logs/minLM_frontier-%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 8
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

# NCCL/ROCm settings
export NCCL_DEBUG=WARN # Reduce noise now that basic comms seem ok
export NCCL_SOCKET_IFNAME=hsn0 # Use primary HSN interface
export NCCL_NET_GDR_LEVEL=3
export RCCL_DEBUG=WARN # Reduce noise
export FI_CXI_ATS=0
export FI_LOG_LEVEL=WARN # Reduce noise
export PDSH_RCMD_TYPE=ssh  # Required for DeepSpeed launcher with SSH

# Setup hostfile for DeepSpeed launcher
HOSTS_PATH=./hosts-job$SLURM_JOB_ID
HOSTFILE_PATH=./hostfile-job$SLURM_JOB_ID.txt
scontrol show hostnames $SLURM_NODELIST > $HOSTS_PATH
rm -f $HOSTFILE_PATH # Ensure clean hostfile
# Create hostfile with node names and slots (GPUs per node)
while IFS= read -r host; do
  echo "$host slots=8" >> $HOSTFILE_PATH
done < $HOSTS_PATH
echo "Hostfile created at $HOSTFILE_PATH"
cat $HOSTFILE_PATH # Print hostfile content for verification

# ---- CRITICAL: Create .deepspeed_env file ----
# This ensures necessary environment variables are propagated to all nodes
echo "Creating .deepspeed_env file..."
echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
# Add ROCm path - Check which variable is set by module load
if [ -n "$ROCM_PATH" ]; then
  echo "ROCM_PATH=$ROCM_PATH" >> .deepspeed_env
elif [ -n "$ROCM_HOME" ]; then
  echo "ROCM_HOME=$ROCM_HOME" >> .deepspeed_env
else
  # Fallback to typical ROCm path on Frontier if variables not set
  echo "WARNING: Neither ROCM_PATH nor ROCM_HOME set, using default path"
  echo "ROCM_PATH=/opt/rocm-6.2.4" >> .deepspeed_env
fi
# Add other environment variables needed by DeepSpeed/PyTorch
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" >> .deepspeed_env
echo "HF_HOME=$HF_HOME" >> .deepspeed_env
# Only include essential variables to reduce log noise
# echo "NCCL_DEBUG=$NCCL_DEBUG" >> .deepspeed_env
# echo "RCCL_DEBUG=$RCCL_DEBUG" >> .deepspeed_env
# echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" >> .deepspeed_env
# echo "NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL" >> .deepspeed_env
echo ".deepspeed_env created with ROCm paths for op builder"
cat .deepspeed_env # Show content for debugging

# Ensure MASTER_PORT is propagated to all processes
export UCX_TLS=rc,tcp,sm
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export TORCH_DISTRIBUTED_DEBUG=INFO # Valid values are OFF, INFO, or DETAIL

# --- Generate Timestamped Output Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}" # Place timestamped runs inside ./outputs
echo "Generated Output Directory: ${OUTPUT_DIR}"
# Note: Rank 0 in train.py will create this directory

# Calculate ranks
ranks_per_node=8
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))
echo "Total ranks: $ranks_total"

# Create base log dir (timestamped run dir created by rank 0 in script)
mkdir -p logs
mkdir -p ./outputs # Ensure the base ./outputs directory exists

# Launch with DeepSpeed - using .deepspeed_env for environment propagation
echo "Starting DeepSpeed launcher with ROCm environment..."
deepspeed --hostfile=$HOSTFILE_PATH --master_port=3442 train.py \
   --data /lustre/orion/scratch/erikgarrison/bif148/enwik8.txt \
   --output "$OUTPUT_DIR" \
   --train_steps 10000 \
   --validate_every 200 \
   --save_every 500 \
   --lr 1e-4 \
   --batch_size 16 \
   --grad_accum 1 \
   --seq_len 2048 \
   --params 100m \
   --tp_size 8 \
   --keep_checkpoints 5 \
   --deepspeed \
   --deepspeed_config ds_config.json

echo "DeepSpeed launcher finished."
# Clean up temporary files
rm -f $HOSTS_PATH $HOSTFILE_PATH .deepspeed_env

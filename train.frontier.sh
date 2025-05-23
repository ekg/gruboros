#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_frontier
#SBATCH -o logs/minLM_frontier-%j.out
#SBATCH -e logs/minLM_frontier-%j.err
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 128                # Number of nodes
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
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed_extensions
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=1

# Get the first node of the allocation
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Get the IP address of the hsn0 interface on the MASTER_NODE_HOSTNAME
# This command is run ON THE MASTER_NODE_HOSTNAME via srun
export MASTER_ADDR=$(srun --ntasks=1 --nodes=1 -w "$MASTER_NODE_HOSTNAME" ip -4 addr show hsn0 | grep -oP 'inet \K[\d.]+')
if [ -z "$MASTER_ADDR" ]; then
  echo "Failed to get MASTER_ADDR for hsn0. Trying hsn1..."
  export MASTER_ADDR=$(srun --ntasks=1 --nodes=1 -w "$MASTER_NODE_HOSTNAME" ip -4 addr show hsn1 | grep -oP 'inet \K[\d.]+')
fi
if [ -z "$MASTER_ADDR" ]; then
  echo "CRITICAL: Failed to determine MASTER_ADDR. Exiting."
  exit 1
fi
echo "Using MASTER_NODE_HOSTNAME: $MASTER_NODE_HOSTNAME"
echo "Determined MASTER_ADDR (IP of hsn0/hsn1 on master): $MASTER_ADDR"

export MASTER_PORT=3442
export TORCH_DISTRIBUTED_TIMEOUT=3600s    # 1 hr timeout for initialization

# NCCL/RCCL settings optimized for Frontier's Slingshot fabric
export UCX_TLS=rc,sm
export NCCL_DEBUG=INFO                     # Set to INFO to diagnose distribution issues
export RCCL_DEBUG=INFO                     # ROCm-specific debug info
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1         # Both error flags for redundancy
export NCCL_TIMEOUT=1800000                # 30 min timeout in ms (was 10000000)
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=hsn0             # Primary high-speed network interface
export NCCL_CROSS_NIC=1                    # Enable multi-rail
export NCCL_NET_GDR_LEVEL=3                # RDMA via OFI
export NCCL_MIN_NCHANNELS=32               # MI250X tuning
export NCCL_NTHREADS=4                     # Thread tuning for NCCL
export NCCL_NSOCKS_PERTHREAD=4             # Socket tuning for NCCL
export NCCL_BUFFSIZE=2097152               # 2MB buffer size
export NCCL_CUMEM_ENABLE=0                 # Disable CUDA unified memory

# Performance tuning
export FI_CXI_ATS=0
export NCCL_COLLNET_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=1M
export NCCL_P2P_PCI_RELAXED_ORDERING=1

# MIOPEN cache paths per node to avoid contention
export MIOPEN_USER_DB_PATH="/tmp/${USER:-user}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH" # Ensure the directory exists

# Create hostfile for DeepSpeed from Slurm allocation
HOSTFILE_NAME="hostfile-job$SLURM_JOB_ID.txt"
HOSTFILE_PATH="$PWD/$HOSTFILE_NAME" # Use full path
GPUS_PER_NODE=8

# Get the list of nodes from Slurm
scontrol show hostnames $SLURM_JOB_NODELIST > "$PWD/slurm_hosts-job$SLURM_JOB_ID.txt"

# Create a proper hostfile with slots information
rm -f "$HOSTFILE_PATH"  # Remove existing hostfile if present
while IFS= read -r host; do
    echo "$host slots=$GPUS_PER_NODE" >> "$HOSTFILE_PATH"
done < "$PWD/slurm_hosts-job$SLURM_JOB_ID.txt"

echo "Created hostfile with contents:"
cat "$HOSTFILE_PATH"

# Generate timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1b_tweak"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
echo "Generated Output Directory: ${OUTPUT_DIR}"

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

# Print SLURM environment for debugging
env | grep SLURM

# COMBINED APPROACH: Using both hostfile and explicit parameters
echo "Starting DeepSpeed with combined approach (hostfile + direct parameters)..."
deepspeed \
  --hostfile="$HOSTFILE_PATH" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --data "$DATA" \
  --output "$OUTPUT_DIR" \
  --train_steps 100000 \
  --validate_every 256 \
  --save_every 256 \
  --lr 0.018 \
  --sf_beta 0.9 \
  --weight_decay 0.001 \
  --batch_size 1 \
  --grad_accum 32 \
  --gradient_clipping 1.0 \
  --seq_len 2048 \
  --params 1g \
  --tp_size 8 \
  --keep_checkpoints 5 \
  --deepspeed \
  --deepspeed_config ds_config.json \
  --rocm

echo "Training finished."

# Clean up temporary files
rm -f "$PWD/slurm_hosts-job$SLURM_JOB_ID.txt" "$HOSTFILE_PATH"

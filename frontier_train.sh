#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_pmi              # Job name with PMI hybrid parallelism
#SBATCH -o %x-%j.out              # Output file name (%x=job name, %j=job id)
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 01:00:00               # Maximum job time (HH:MM:SS)
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 4                      # Request 4 nodes for multi-node training
#SBATCH --ntasks-per-node=8       # 8 tasks per node (1 per GPU)
#SBATCH --gpus-per-node=8         # Request all 8 GPUs on each node
#SBATCH --exclusive               # Request exclusive access to node

# Uncomment to set specific node features if needed
##SBATCH --constraint=<feature>   # Use specific node features

#--------------------------------------
# Job setup - minimal environment setup
#--------------------------------------

# Only load modules needed for job management
module load PrgEnv-amd

# Setup output directory with date and run info
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/lustre/orion/scratch/erikgarrison/bif148/gruboros/run_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Display information about the job
echo "========== Job Information =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES ($(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' '))"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Output directory: $OUTPUT_DIR"
echo "======================================"

#--------------------------------------
# Run the training
#--------------------------------------

# Define common training parameters
DATA_PATH="/lustre/orion/scratch/erikgarrison/bif148/enwik8.txt"
MODEL_SIZE="512m"        # Target model size (could be 15m, 100m, 1g, etc.)
SEQ_LEN="2k"             # Sequence length
BATCH_SIZE="4"           # Batch size per GPU

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * SLURM_GPUS_PER_NODE * SLURM_NNODES))
echo "Running with effective batch size: $EFFECTIVE_BATCH across $SLURM_NNODES nodes with $SLURM_GPUS_PER_NODE GPUs each"

# Make the wrapper script executable
chmod +x ./run_deepspeed.sh

# Load required modules for Frontier
module load PrgEnv-amd
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Set environment variables for Frontier high-speed network
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export NCCL_TREE_THRESHOLD=0   # Optimize for multi-node
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=16
export NCCL_NET_PLUGIN=ucx
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_IB_HCA=hsn0,hsn1,hsn2,hsn3

# ROCm settings
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export ROCR_LOG_LEVEL=INFO
export MPICH_GPU_SUPPORT_ENABLED=1

# MIOpen cache setup
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_USER_DB_PATH}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p $MIOPEN_USER_DB_PATH

# Set PMI environment variables for improved reliability
export FI_CXI_RDZV_PROTO=alt_read

# Using PMI for process coordination (no need for MASTER_ADDR/PORT)
echo "Using Slurm PMI for coordinating $SLURM_NTASKS processes across $SLURM_NNODES nodes"

# Set up micromamba environment
export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate gruboros

# Launch with srun using PMI directly
srun --mpi=pmi2 python train.py \
    --data $DATA_PATH \
    --params $MODEL_SIZE \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accum 32 \
    --output $OUTPUT_DIR \
    --tp_size 8 \
    --ddp_size $SLURM_NNODES \
    --train_steps 10000 \
    --validate_every 200 \
    --save_every 200 \
    --keep_checkpoints 5 \
    --log_sample_hashes

echo "Training complete. Results saved to $OUTPUT_DIR"

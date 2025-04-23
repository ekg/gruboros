#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_train            # Job name
#SBATCH -o %x-%j.out              # Output file name (%x=job name, %j=job id)
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 0:10:00                # Maximum job time (HH:MM:SS)
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 1                      # Number of nodes (increase for multi-node)
#SBATCH --ntasks-per-node=8       # Number of GPUs per node (max 8 on Frontier)
#SBATCH --gpus-per-node=8         # Request all 8 GPUs on the node
#SBATCH --exclusive               # Request exclusive access to node

# Uncomment to set specific node features if needed
##SBATCH --constraint=<feature>   # Use specific node features

#--------------------------------------
# Environment setup
#--------------------------------------

# Load necessary modules
module load PrgEnv-amd
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# set up env
micromamba activate gruboros

# Print loaded modules and environment info for debugging
module list
env | grep ROCM
env | grep SLURM

# Enable detailed ROCm logging and setup
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export HSA_TOOLS_LIB=1
export ROCR_LOG_LEVEL=INFO

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Prevent file system contention with unique directories
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_USER_DB_PATH}"
mkdir -p $MIOPEN_USER_DB_PATH

# Enhanced network settings for Frontier Slingshot network
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export NCCL_DEBUG=INFO

# Setup output directory with date and run info
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./outputs/run_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Display information about the job
echo "========== Job Information =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Output directory: $OUTPUT_DIR"
echo "======================================"

# Print GPU information
rocm-smi --showdriverversion
rocm-smi --showproductname
rocm-smi

#--------------------------------------
# Run the training
#--------------------------------------

# Define common training parameters
DATA_PATH="/autofs/nccs-svm1_home1/erikgarrison/gruboros/enwik8.txt"
MODEL_SIZE="100m"        # Target model size (could be 15m, 100m, 1g, etc.)
SEQ_LEN="2k"             # Sequence length
BATCH_SIZE="32"          # Batch size per GPU

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * SLURM_NTASKS))
echo "Running with effective batch size: $EFFECTIVE_BATCH across $SLURM_NTASKS GPUs"

# Make the wrapper script executable
chmod +x ./run_deepspeed.sh

# Launch training using srun with the wrapper script (Recommended for Frontier)
srun -N $SLURM_NNODES -n $SLURM_NTASKS --gpus-per-node=$SLURM_GPUS_PER_NODE \
    ./run_deepspeed.sh \
    --data $DATA_PATH \
    --params $MODEL_SIZE \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accum 1 \
    --output $OUTPUT_DIR \
    --tp_size 1 \
    --train_steps 10000 \
    --validate_every 200 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --log_sample_hashes

# Alternative distributed launch for simple single-node using DeepSpeed (if needed)
# deepspeed --num_gpus=$SLURM_GPUS_PER_NODE train.py [...parameters...]

echo "Training complete. Results saved to $OUTPUT_DIR"

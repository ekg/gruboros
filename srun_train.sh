#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_train            # Job name
#SBATCH -o %x-%j.out              # Output file name
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 1:00:00                # Maximum job time
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 2                      # Number of nodes
#SBATCH --ntasks-per-node=8       # 8 tasks per node (one per GPU)
#SBATCH --gpus-per-node=8         # All 8 GPUs per node
#SBATCH --exclusive               # Exclusive node access

# Essential environment setup with error handling
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
export ROCM_HOME=/opt/rocm-6.2.4

# Verify ROCM is available
if [ ! -d "$ROCM_HOME" ]; then
  echo "ERROR: ROCm directory not found at $ROCM_HOME"
  echo "Loading module may have failed. Check module system."
fi

# Micromamba activation with better error handling
export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba'

if [ ! -f "$MAMBA_EXE" ]; then
  echo "ERROR: micromamba executable not found at $MAMBA_EXE"
  exit 1
fi

# Create a temporary hook script and source it directly (more reliable)
HOOK_SCRIPT=$(mktemp)
"$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" > "$HOOK_SCRIPT"
source "$HOOK_SCRIPT"
rm -f "$HOOK_SCRIPT"

# Activate with explicit error checking
echo "Activating micromamba environment 'gruboros'..."
micromamba activate gruboros || (echo "Failed to activate environment directly, trying alternative method" && \
  eval "$("$MAMBA_EXE" shell init --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" && \
  micromamba activate gruboros)

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found after activation. Trying direct path execution..."
    export PATH="/lustre/orion/bif148/scratch/erikgarrison/micromamba/envs/gruboros/bin:$PATH"
fi
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate gruboros environment"
  echo "Available environments:"
  micromamba env list
  exit 1
fi

# Verify we have python in path
which python
if [ $? -ne 0 ]; then
  echo "ERROR: Python not found in path after environment activation"
  echo "PATH: $PATH"
  exit 1
fi

# Critical ROCm + Network settings
export LD_LIBRARY_PATH=$ROCM_HOME/rccl/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=3
export FI_LOG_LEVEL=info
export OMP_NUM_THREADS=2

# Find master node for distributed communication - more robust approach
if [ -z "$SLURM_NODELIST" ]; then
  echo "ERROR: SLURM_NODELIST is empty. Are you running under SLURM?"
  exit 1
fi

# Create node list file
NODE_LIST_FILE="job.node.list"
scontrol show hostnames $SLURM_NODELIST > $NODE_LIST_FILE

# Check if the file has content
if [ ! -s "$NODE_LIST_FILE" ]; then
  echo "ERROR: Failed to get node list from SLURM"
  # Fallback to local hostname if no nodes listed
  hostname > $NODE_LIST_FILE
fi

# Get first node in the list
first=$(head -n 1 $NODE_LIST_FILE)
echo "First node in allocation: $first"

# Directly use the hostname as MASTER_ADDR (more reliable than trying to get IP)
export MASTER_ADDR=$first
export MASTER_PORT=29500

echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"

# Setup output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$MEMBERWORK/bif148/gruboros/run_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Calculate ranks with safety checks
if [ -z "$SLURM_NNODES" ]; then
  echo "ERROR: SLURM_NNODES is not set. Defaulting to 1."
  SLURM_NNODES=1
fi

RANKS_PER_NODE=8
TOTAL_RANKS=$((SLURM_NNODES * RANKS_PER_NODE))

# Make sure we have at least one task
if [ "$TOTAL_RANKS" -lt 1 ]; then
  echo "ERROR: Invalid task count ($TOTAL_RANKS). Setting to 8."
  TOTAL_RANKS=8
fi

echo "Calculated $TOTAL_RANKS total ranks ($SLURM_NNODES nodes × $RANKS_PER_NODE ranks per node)"

# Training parameters
DATA_PATH="/lustre/orion/scratch/erikgarrison/bif148/enwik8.txt"
MODEL_SIZE="100m"
SEQ_LEN="2k"
BATCH_SIZE="32"
GRAD_ACCUM=1

echo "Running with $TOTAL_RANKS total GPUs across $SLURM_NNODES nodes"
echo "Output directory: $OUTPUT_DIR"

# Launch training with srun directly (based on working examples)
srun -u -n$TOTAL_RANKS -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
    python train.py \
    --data $DATA_PATH \
    --params $MODEL_SIZE \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --output $OUTPUT_DIR \
    --train_steps 1000 \
    --validate_every 200 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --port $MASTER_PORT \
    --deepspeed \
    --deepspeed_config ds_config.json

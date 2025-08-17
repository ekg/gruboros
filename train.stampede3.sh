#!/bin/bash
#SBATCH -J gruboros_350m_4node    # Job name
#SBATCH -o logs/gruboros_%j.out   # Output file (%j expands to jobID)
#SBATCH -e logs/gruboros_%j.err   # Error file
#SBATCH -p h100                   # H100 partition on Stampede3
#SBATCH -N 4                      # Number of nodes
#SBATCH --ntasks-per-node=4       # Tasks per node - should match GPU count (dynamically verified below)
#SBATCH -t 24:00:00              # Time limit (24 hours)

set -e -x

# --- Module Loading ---
module load gcc/13.2.0
module load cuda/12.8
module load python/3.9.18
module list

# --- Environment Setup ---
# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros

# Ensure scipy is installed
micromamba install -y scipy -c conda-forge || pip install scipy

# --- Distributed Settings ---
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500

# Stampede3 H100 nodes have 4 GPUs per node
GPUS_PER_NODE=4
echo "Using $GPUS_PER_NODE GPUs per node (Stampede3 H100 configuration)"

export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
# Don't set RANK and LOCAL_RANK here - srun will set them per task
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
export RANKS_PER_NODE=$GPUS_PER_NODE
export TORCH_DISTRIBUTED_TIMEOUT=7200s  # 2 hour timeout for large models
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO  # For debugging, remove in production

# --- File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="350m_enwik9_4node_1k_batch4"

# Get git commit hash
GIT_HASH=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")
fi

# Build output directory name
if [ -n "$GIT_HASH" ]; then
    OUTPUT_DIR="$SCRATCH/models/gruboros/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    OUTPUT_DIR="$SCRATCH/models/gruboros/${TIMESTAMP}_${NAME}"
fi

# Data path
DATA_PATH="$SCRATCH/train/enwik9"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    echo "Please download enwik9:"
    echo "  mkdir -p $SCRATCH/train"
    echo "  cd $SCRATCH/train"
    echo "  wget http://mattmahoney.net/dc/enwik9.zip"
    echo "  unzip enwik9.zip"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"/{gossip,metrics,logs}
mkdir -p logs  # For SLURM logs

# Gossip temp directory (node-local /tmp is fast)
GOSSIP_TEMP_DIR="/tmp/gossip_${SLURM_JOB_ID}"
mkdir -p "$GOSSIP_TEMP_DIR"

# --- Log System Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Master addr: $MASTER_ADDR"
echo "Output dir: $OUTPUT_DIR"
echo "Data path: $DATA_PATH"

# --- Launch Training ---
echo "Starting 350M parameter pure RNN training on 4 H100 nodes with 1k chunk size, batch_size=4"

# Use srun to launch on all allocated resources
srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=$GPUS_PER_NODE \
     --cpus-per-task=$((96 / GPUS_PER_NODE)) \
     --distribution=block:block \
     ./run_training_task.sh \
     --data "$DATA_PATH" \
     --output "$OUTPUT_DIR" \
     --params 350m \
     --dim 1024 \
     --expansion_factor 4.0 \
     --ff_mult 0 \
     --train_steps 10000000 \
     --save_every 500 \
     --lr 0.001 \
     --sf_beta 0.9 \
     --sf_beta2 0.995 \
     --weight_decay 0.0001 \
     --batch_size 4 \
     --grad_accum 16 \
     --chunk_size 1024 \
     --keep_checkpoints 5 \
     --keep_elite 32 \
     --archive_rate 0.0067 \
     --gossip_merge_method recombination \
     --gossip_recombination_alpha 0.2 \
     --gossip_optimizer_recombination interpolate \
     --gossip_mixing_rate 0.002 \
     --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
     --gossip_p_value_threshold 0.1 \
     --gossip_fitness_window 10000 \
     --validation_sequences 16 \
     --validation_sequence_length 8k \
     --validation_interval 500 \
     --filesystem-coordinator \
     --fitness-weighted-checkpointing \
     --elite-checkpoint-multiplier 20.0 \
     --cuda

echo "Training completed at $(date)"

# --- Cleanup ---
rm -rf "$GOSSIP_TEMP_DIR"
echo "Cleaned up temporary directories"

# --- Final Model Location ---
echo "Models saved to: $OUTPUT_DIR"
echo "Latest checkpoint: $OUTPUT_DIR/latest.pt"
echo "Best checkpoint: $OUTPUT_DIR/best.pt"

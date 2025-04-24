#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_train            # Job name
#SBATCH -o %x-%j.out              # Output file name (%x=job name, %j=job id)
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 01:00:00               # Maximum job time (HH:MM:SS)
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 4                      # Request 4 nodes for multi-node training
#SBATCH --ntasks-per-node=1       # One task per node for DDP
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
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
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

# Set recommended fixed port for distributed communication
export MASTER_PORT=3442

# Launch training using srun with the wrapper script
srun -N $SLURM_NNODES -n $SLURM_NTASKS --gpus-per-node=$SLURM_GPUS_PER_NODE \
    ./run_deepspeed.sh \
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
    --log_sample_hashes \
    --port $MASTER_PORT

echo "Training complete. Results saved to $OUTPUT_DIR"

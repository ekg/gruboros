#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_train            # Job name
#SBATCH -o %x-%j.out              # Output file name (%x=job name, %j=job id)
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 1:00:00                # Maximum job time (HH:MM:SS) - increased for multi-node
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 2                      # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=8       # Launch 8 tasks per node (one per GPU)
#SBATCH --gpus-per-node=8         # Request all 8 GPUs on each node
#SBATCH --exclusive               # Request exclusive access to node

# Calculate total tasks and ranks per node
RANKS_PER_NODE=8
TOTAL_RANKS=$((SLURM_NNODES * RANKS_PER_NODE))

#--------------------------------------
# Job setup - minimal environment setup
#--------------------------------------

# Only load modules needed for job management
module load PrgEnv-amd

# Setup output directory with date and run info
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$MEMBERWORK/bif148/gruboros/run_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Create hostfile for DeepSpeed - each node should appear only once
HOSTFILE=hostfile.txt
# Get unique hostnames using scontrol
scontrol show hostnames $SLURM_JOB_NODELIST > unique_hosts_$SLURM_JOB_ID
# Add slot information - each host gets 8 slots (for 8 GPUs)
sed 's/$/ slots=8/' unique_hosts_$SLURM_JOB_ID > $HOSTFILE
# Clean up temporary file
rm -f unique_hosts_$SLURM_JOB_ID

# Set ROCm environment variables for DeepSpeed
export CUDA_HOME=/opt/rocm-6.2.4
export HIP_CLANG_PATH=/opt/rocm-6.2.4/llvm
export ROCM_HOME=/opt/rocm-6.2.4
export DS_BUILD_OPS=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

# Display information about the job
echo "========== Job Information =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Output directory: $OUTPUT_DIR"
echo "Hostfile: $HOSTFILE"
echo "ROCm version: $(rocm-smi --showdriverversion 2>/dev/null || echo 'not available')"
echo "======================================"

#--------------------------------------
# Run the training
#--------------------------------------

# Define common training parameters
DATA_PATH="/lustre/orion/scratch/erikgarrison/bif148/enwik8.txt"
MODEL_SIZE="100m"        # Target model size (could be 15m, 100m, 1g, etc.)
SEQ_LEN="2k"             # Sequence length
BATCH_SIZE="32"          # Batch size per GPU
GRAD_ACCUM=1             # Gradient accumulation steps

# Calculate effective batch size across all GPUs
EFFECTIVE_BATCH=$((BATCH_SIZE * TOTAL_RANKS * GRAD_ACCUM))
echo "Running with effective batch size: $EFFECTIVE_BATCH across $TOTAL_RANKS GPUs ($SLURM_NNODES nodes)"

# Make the wrapper script executable
chmod +x ./run_deepspeed.sh

# Set fixed port for distributed communication - use specific port to avoid conflicts
export MASTER_PORT=3442

# Set critical environment variables for all processes
export CUDA_HOME=/opt/rocm-6.2.4
export HIP_CLANG_PATH=/opt/rocm-6.2.4/llvm
export ROCM_HOME=/opt/rocm-6.2.4
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed_rocm_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# We'll try both approaches - first the direct srun launch (more reliable for environment vars)
srun -u -n$TOTAL_RANKS -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
    ./run_deepspeed.sh python train.py \
    --data $DATA_PATH \
    --params $MODEL_SIZE \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --output $OUTPUT_DIR \
    --tp_size 1 \
    --train_steps 10000 \
    --validate_every 200 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --log_sample_hashes \
    --port $MASTER_PORT \
    --deepspeed \
    --deepspeed_config ds_config.json

echo "Training complete. Results saved to $OUTPUT_DIR"

# Option for srun-based launch (uncomment to use instead of DeepSpeed launcher)
# srun -u -n$TOTAL_RANKS -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
#     python train.py \
#     --data $DATA_PATH \
#     --params $MODEL_SIZE \
#     --seq_len $SEQ_LEN \
#     --batch_size $BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --output $OUTPUT_DIR \
#     --tp_size 1 \
#     --train_steps 10000 \
#     --validate_every 200 \
#     --save_every 500 \
#     --keep_checkpoints 5 \
#     --log_sample_hashes \
#     --port $MASTER_PORT \
#     --deepspeed \
#     --deepspeed_config ds_config.json

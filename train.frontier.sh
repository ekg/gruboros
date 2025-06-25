#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_gossip_srun_gloo
#SBATCH -o logs/minLM_gossip-%j.out
#SBATCH -e logs/minLM_gossip-%j.err
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1  # Important: Binds one task to one GPU for stability
#SBATCH -C nvme

set -x

# --- Environment Setup ---
# Your existing environment setup is correct.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros
module load PrgEnv-gnu gcc/11.2.0 rocm/6.2.4 craype-accel-amd-gfx90a

# --- Distributed Settings for srun + gloo ---
export OMP_NUM_THREADS=1
export RANKS_PER_NODE=$SLURM_NTASKS_PER_NODE

# 1. Set the Master Address: srun needs this for the gloo backend to rendezvous.
#    This is the canonical way to get the head node's hostname in a SLURM job.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# 2. Set the Master Port: Any free port will do.
export MASTER_PORT=3442

# 3. Critical for Performance: Tell gloo to use the High-Speed Network Interface.
export GLOO_SOCKET_IFNAME=hsn0

# --- Paths and Directories (Unchanged) ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_32k_srun_gloo"
GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")

if [ -n "$GIT_HASH" ]; then
    OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}"
fi
DATA="/lustre/orion/bif148/scratch/erikgarrison/fineweb-edu/sample/350BT.txt"
GOSSIP_TEMP_DIR="/mnt/bb/$(whoami)/gossip_temp/${SLURM_JOB_ID}"

# --- Pre-create Directories (Unchanged, this is good practice) ---
mkdir -p logs "${OUTPUT_DIR}"/gossip "${OUTPUT_DIR}"/metrics
# Use srun to create the temp directory on every allocated node
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $GOSSIP_TEMP_DIR"
echo "Created gossip temp directory on all node-local NVMe drives: $GOSSIP_TEMP_DIR"

# --- Launch Training with srun ---
echo "Starting Filesystem-Augmented Hybrid Evolution with srun launcher and the Gloo backend."
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"

# srun will start 128 total tasks (16 nodes * 8 tasks/node).
# It automatically provides RANK, WORLD_SIZE, and LOCAL_RANK to each process.
# Your Python script already reads these variables, so it will work seamlessly.
# The --cpu-bind flag is highly recommended for performance and stability.
srun --cpu-bind=verbose,map_cpu:49,57,17,25,1,9,33,41 python train.py \
  --data "$DATA" \
  --output "$OUTPUT_DIR" \
  --train_steps 100000 \
  --save_every 100 \
  --lr 0.005 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 16 \
  --chunk_size 2048 \
  --context_chunks 16 \
  --dim 2048 \
  --params 1g \
  --keep_checkpoints 3 \
  --keep_elite 10 \
  --archive_rate 0.1 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.5 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.001 \
  --gossip_fitness_decay 0.995 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --gossip-node-local-lock \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 4.0 \
  --rejuvenation-threshold 0.75 \
  --rejuvenation-probability 0.002 \
  --rocm

echo "Training finished."
# --- Cleanup (Unchanged) ---
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf $GOSSIP_TEMP_DIR"
echo "Cleaned up gossip temp directories from all node-local NVMe drives."

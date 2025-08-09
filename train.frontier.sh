#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J llm_gossip
#SBATCH -o logs/minLM_gossip-%j.out
#SBATCH -e logs/minLM_gossip-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 64
#SBATCH -q debug
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH -C nvme

set -x

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Environment Setup ---
# Your existing environment setup is correct.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros
module load PrgEnv-gnu gcc/11.2.0 rocm/6.2.4 craype-accel-amd-gfx90a

# --- Environment Settings ---
export OMP_NUM_THREADS=1
export RANKS_PER_NODE=$SLURM_NTASKS_PER_NODE

# --- Paths and Directories (Unchanged) ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_2k"
GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")

if [ -n "$GIT_HASH" ]; then
    export OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    export OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}"
fi
export DATA="/lustre/orion/bif148/scratch/erikgarrison/commonpile/commapile.txt"
export RESUME_FROM="/lustre/orion/bif148/scratch/erikgarrison/gruboros.test.1/resume/latest.pt"
export GOSSIP_TEMP_DIR="/mnt/bb/$(whoami)/gossip_temp/${SLURM_JOB_ID}"

# --- Pre-create Directories (Unchanged, this is good practice) ---
mkdir -p logs "${OUTPUT_DIR}"/gossip "${OUTPUT_DIR}"/metrics
# Use srun to create the temp directory on every allocated node
srun --no-kill --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $GOSSIP_TEMP_DIR"
echo "Created gossip temp directory on all node-local NVMe drives: $GOSSIP_TEMP_DIR"

# --- Launch Training with srun ---
echo "Starting Filesystem-Augmented Hybrid Evolution with srun launcher and the Gloo backend."
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"

# --- FIX: Use double quotes and escape SLURM vars ---
# This allows script variables like $OUTPUT_DIR to be expanded by the main shell,
# while SLURM variables like \$SLURM_PROCID are expanded by srun for each task.
srun --no-kill --cpu-bind=verbose,map_cpu:49,57,17,25,1,9,33,41 bash -c "
export RANK=\$SLURM_PROCID
export WORLD_SIZE=\$SLURM_NPROCS
export LOCAL_RANK=\$SLURM_LOCALID

# Execute the python script with the correct environment now set
( python train.py \
  --data \"$DATA\" \
  --output \"$OUTPUT_DIR\" \
  --resume \"$RESUME_FROM\" \
  --params 1g \
  --dim 1536 \
  --expansion_factor 3.0 \
  --ff_mult 1.5 \
  --train_steps 10000000 \
  --save_every 50 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --grad_accum 1 \
  --chunk_size 2048 \
  --keep_checkpoints 5 \
  --keep_elite 16 \
  --archive_rate 0.05 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.3 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.002 \
  --gossip_temp_dir \"$GOSSIP_TEMP_DIR\" \
  --gossip_fitness_window 10000 \
  --gossip-node-local-lock \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 10.0 \
  --rocm ) || (
    echo \"[\$(date)] Rank \$SLURM_PROCID died on \$(hostname)\" >> \"$OUTPUT_DIR/deaths.log\"
    sleep infinity
  )
"

echo "Training finished."

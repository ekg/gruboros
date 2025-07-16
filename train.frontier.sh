#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_gossip_srun_gloo
#SBATCH -o logs/minLM_gossip-%j.out
#SBATCH -e logs/minLM_gossip-%j.err
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -N 64
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
    export OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    export OUTPUT_DIR="./outputs/${TIMESTAMP}_${NAME}"
fi
export DATA="/lustre/orion/bif148/scratch/erikgarrison/commonpile/commapile.txt"
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
  --params 1g \
  --dim 1536 \
  --expansion_factor 3.0 \
  --ff_mult 1.5 \
  --train_steps 10000000 \
  --save_every 500 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --grad_accum 8 \
  --chunk_size 2048 \
  --context_chunks 8 \
  --keep_checkpoints 5 \
  --keep_elite 32 \
  --archive_rate 0.2 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.3 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.002 \
  --gossip_temp_dir \"$GOSSIP_TEMP_DIR\" \
  --gossip_fitness_window 1000 \
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

#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_gossip
#SBATCH -o logs/minLM_gossip-%j.out
#SBATCH -e logs/minLM_gossip-%j.err
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 128
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -q debug

set -x

# --- Environment Setup ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros
export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu gcc/11.2.0 rocm/6.2.4 craype-accel-amd-gfx90a

# --- Distributed Settings for Launcher & Script ---
export OMP_NUM_THREADS=1
export RANKS_PER_NODE=$SLURM_NTASKS_PER_NODE
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --ntasks=1 --nodes=1 -w "$MASTER_NODE_HOSTNAME" ip -4 addr show hsn0 | grep -oP 'inet \K[\d.]+')
export MASTER_PORT=3442
export TORCH_DISTRIBUTED_TIMEOUT=7200s
# Use GLOO for peer discovery, our gossip protocol uses its own TCP sockets.
export TORCH_DISTRIBUTED_BACKEND="gloo"
echo "Using GLOO backend for initial process group."

# --- Create Hostfile for DeepSpeed LAUNCHER ---
HOSTFILE_NAME="hostfile-job$SLURM_JOB_ID.txt"
scontrol show hostnames $SLURM_JOB_NODELIST | while IFS= read -r host; do echo "$host slots=$RANKS_PER_NODE"; done > "$HOSTFILE_NAME"
echo "Launcher hostfile created at $HOSTFILE_NAME"

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="100m_pure_gossip"
OUTPUT_DIR="./outputs/gruboros_${TIMESTAMP}_${NAME}"
DATA="/lustre/orion/bif148/scratch/erikgarrison/enwik8"
mkdir -p logs "$OUTPUT_DIR"

# --- Launch Training ---
echo "Starting Pure Gossip training. DeepSpeed is used ONLY as a launcher."
deepspeed \
  --hostfile="$HOSTFILE_NAME" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --data "$DATA" \
  --output "$OUTPUT_DIR" \
  --train_steps 100k \
  --validate_every 200 \
  --save_every 200 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 4 \
  --grad_accum 1 \
  --seq_len 2k \
  --params 100m \
  --keep_checkpoints 5 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.5 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.05 \
  --rocm

echo "Training finished."
rm -f "$HOSTFILE_NAME"

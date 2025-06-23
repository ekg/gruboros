#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_gossip_1G_16k
#SBATCH -o logs/minLM_gossip-%j.out
#SBATCH -e logs/minLM_gossip-%j.err
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -q debug
#SBATCH -C nvme

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
export TORCH_DISTRIBUTED_BACKEND="gloo"
export GLOO_SOCKET_IFNAME=hsn0
echo "Using GLOO backend for initial process group."

# --- Create Hostfile for DeepSpeed LAUNCHER ---
HOSTFILE_NAME="hostfile-job$SLURM_JOB_ID.txt"
scontrol show hostnames $SLURM_JOB_NODELIST | while IFS= read -r host; do echo "$host slots=$RANKS_PER_NODE"; done > "$HOSTFILE_NAME"
echo "Launcher hostfile created at $HOSTFILE_NAME"

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME="1g_16k_context_final"
OUTPUT_DIR="/lustre/orion/bif148/scratch/$(whoami)/outputs/gruboros_${TIMESTAMP}_${NAME}"
DATA="/lustre/orion/bif148/scratch/erikgarrison/fineweb-edu/sample/350BT.txt"

# Use the node-local NVMe for the temporary gossip directory
# Each rank will use this path. The directory needs to be created on each node.
GOSSIP_TEMP_DIR="/mnt/bb/$(whoami)/gossip_temp/${SLURM_JOB_ID}"

# --- *** FIX: PRE-CREATE ALL DIRECTORIES *** ---
# This eliminates any possible race condition inside the Python script.
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/metrics"
echo "Pre-created all output directories at ${OUTPUT_DIR}"

# Use srun to create the temp directory on the NVMe of *every allocated node*
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $GOSSIP_TEMP_DIR"
echo "Created gossip temp directory on all node-local NVMe drives: $GOSSIP_TEMP_DIR"

# --- Launch Training ---
echo "Starting Pure Gossip training. DeepSpeed is used ONLY as a launcher."
deepspeed \
  --hostfile="$HOSTFILE_NAME" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --data "$DATA" \
  --output "$OUTPUT_DIR" \
  --train_steps 30000 \
  --save_every 20 \
  --lr 0.005 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.0001 \
  --batch_size 1 \
  --grad_accum 1 \
  --chunk_size 2048 \
  --context_chunks 8 \
  --params 1g \
  --keep_checkpoints 5 \
  --keep_elite 10 \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.5 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.03 \
  --gossip_fitness_decay 0.95 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --rocm

echo "Training finished."
# Clean up hostfile and the temporary gossip directories on all nodes
rm -f "$HOSTFILE_NAME"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf $GOSSIP_TEMP_DIR"
echo "Cleaned up gossip temp directories from all node-local NVMe drives."

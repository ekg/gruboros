#!/bin/bash
#SBATCH -A BIF148                  # Your project allocation
#SBATCH -J minLM_frontier          # Job name
#SBATCH -o logs/%x-%j.out          # STDOUT → logs/minLM_frontier-<jobid>.out
#SBATCH -e logs/%x-%j.err          # STDERR → logs/minLM_frontier-<jobid>.err
#SBATCH -t 01:00:00                # Walltime HH:MM:SS
#SBATCH -p batch                   # Queue
#SBATCH -N 2                       # Nodes
#SBATCH -q debug
#SBATCH --ntasks-per-node=8       # One task per GPU
#SBATCH --gpus-per-node=8         # All 8 GPUs
#SBATCH --exclusive                # Exclusive node access

set -euo pipefail
set +x

# Set LD_PRELOAD for necessary libraries
export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

# 1) Load modules
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# 0) Setup Python environment - CRITICAL!
# Set up micromamba environment
export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
micromamba activate gruboros

# 2) ROCm & NCCL tuning for Frontier
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export RCCL_DEBUG=INFO
export FI_CXI_ATS=0
export FI_LOG_LEVEL=info

# 3) Build hostfile & MASTER_ADDR
srun hostname > .hosts.$SLURM_JOB_ID
sed 's/$/ slots=8/' .hosts.$SLURM_JOB_ID > hostfile
MASTER_NODE=$(head -n1 .hosts.$SLURM_JOB_ID)
# Get IP of master node
MASTER_ADDR=$(ssh $MASTER_NODE hostname -i | awk '{print $1}')
export MASTER_ADDR
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=$(( SLURM_NNODES * SLURM_GPUS_PER_NODE ))"

# 4) Create log dir
mkdir -p logs

# 5) Launch
srun --mpi=pmi2 \
     -u \
     -n $(( SLURM_NNODES * SLURM_GPUS_PER_NODE )) \
     -c2 \
     --ntasks-per-node=8 \
     --gpus-per-node=8 \
     --gpu-bind=closest \
     python train.py \
         --data /lustre/orion/scratch/erikgarrison/bif148/enwik8.txt \
         --output ./outputs \
         --train_steps 10000 \
         --validate_every 200 \
         --save_every 500 \
         --batch_size 4 \
         --grad_accum 1 \
         --seq_len 2048 \
         --params 100m \
         --tp_size 1 \
         --keep_checkpoints 5 \
         --port $MASTER_PORT

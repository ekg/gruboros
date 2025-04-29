#!/bin/bash

#SBATCH -A BIF148
#SBATCH -J minLM_frontier
#SBATCH -o logs/minLM_frontier-%j.out
#SBATCH -e logs/minLM_frontier-%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH -q debug

set +x
# Setup Python environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Export settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=2

# NCCL/ROCm settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export RCCL_DEBUG=INFO
export FI_CXI_ATS=0
export FI_LOG_LEVEL=info

# Setup hostfile - this is the exact method used in the working GPT-J example
scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=$(ssh $first hostname -I)
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

# Use a random port to avoid conflicts with other jobs
export MASTER_PORT=$((29500 + $RANDOM % 10000))
echo "MASTER_PORT=" $MASTER_PORT

# Calculate ranks
ranks_per_node=8
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))
echo "Total ranks: $ranks_total"

# Create log dir
mkdir -p logs

# Launch with srun (matches the working GPT-J example pattern)
srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest python train.py \
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
   --deepspeed \
   --deepspeed_config ds_config.json \
   --master_addr=$MASTER_ADDR \
   --port $MASTER_PORT

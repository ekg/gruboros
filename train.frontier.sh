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

# Setup hostfile for DeepSpeed launcher
HOSTS_PATH=./hosts-job$SLURM_JOB_ID
HOSTFILE_PATH=./hostfile-job$SLURM_JOB_ID.txt
scontrol show hostnames $SLURM_NODELIST > $HOSTS_PATH
# Create hostfile with node names and slots (GPUs per node)
while IFS= read -r host; do
  echo "$host slots=8" >> $HOSTFILE_PATH
done < $HOSTS_PATH
echo "Hostfile created at $HOSTFILE_PATH"
cat $HOSTFILE_PATH # Print hostfile content for verification

# DeepSpeed launcher will handle all distributed environment variables
# No need to export MASTER_ADDR, MASTER_PORT, LOCAL_RANK, RANK, WORLD_SIZE
echo "Using DeepSpeed launcher with hostfile to manage distributed setup"

# Ensure MASTER_PORT is propagated to all processes
export UCX_TLS=rc,tcp,sm
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Calculate ranks
ranks_per_node=8
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))
echo "Total ranks: $ranks_total"

# Create log dir
mkdir -p logs

# Launch with DeepSpeed - no need to pass local_rank, DeepSpeed handles it
echo "Starting DeepSpeed launcher..."
deepspeed --hostfile=$HOSTFILE_PATH --master_port=3442 python train.py \
   --data /lustre/orion/scratch/erikgarrison/bif148/enwik8.txt \
   --output ./outputs \
   --train_steps 10000 \
   --validate_every 200 \
   --save_every 500 \
   --batch_size 4 \
   --grad_accum 1 \
   --seq_len 2048 \
   --params 100m \
   --tp_size 8 \
   --keep_checkpoints 5 \
   --deepspeed \
   --deepspeed_config ds_config.json

# Clean up temporary files
rm -f $HOSTS_PATH $HOSTFILE_PATH

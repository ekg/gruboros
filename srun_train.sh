#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J minLM_train            # Job name
#SBATCH -o %x-%j.out              # Output file name
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 1:00:00                # Maximum job time
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH -N 2                      # Number of nodes
#SBATCH --ntasks-per-node=8       # 8 tasks per node (one per GPU)
#SBATCH --gpus-per-node=8         # All 8 GPUs per node
#SBATCH --exclusive               # Exclusive node access

# Essential environment setup
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
export ROCM_HOME=/opt/rocm-6.2.4

# Micromamba activation
export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate gruboros

# Critical ROCm + Network settings
export LD_LIBRARY_PATH=$ROCM_HOME/rccl/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export FI_CXI_ATS=0
export NCCL_NET_GDR_LEVEL=3
export FI_LOG_LEVEL=info
export OMP_NUM_THREADS=2

# Find master node for distributed communication
scontrol show hostnames $SLURM_NODELIST > job.node.list
first=$(head -n 1 job.node.list)
ips=$(ssh $first hostname -I)
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
export MASTER_PORT=29500

echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"

# Setup output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$MEMBERWORK/bif148/gruboros/run_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Calculate ranks
RANKS_PER_NODE=8
TOTAL_RANKS=$((SLURM_NNODES * RANKS_PER_NODE))

# Training parameters
DATA_PATH="/lustre/orion/scratch/erikgarrison/bif148/enwik8.txt"
MODEL_SIZE="100m"
SEQ_LEN="2k"
BATCH_SIZE="32"
GRAD_ACCUM=1

echo "Running with $TOTAL_RANKS total GPUs across $SLURM_NNODES nodes"
echo "Output directory: $OUTPUT_DIR"

# Launch training with srun directly (based on working examples)
srun -u -n$TOTAL_RANKS -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
    python train.py \
    --data $DATA_PATH \
    --params $MODEL_SIZE \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --output $OUTPUT_DIR \
    --train_steps 1000 \
    --validate_every 200 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --port $MASTER_PORT \
    --deepspeed \
    --deepspeed_config ds_config.json

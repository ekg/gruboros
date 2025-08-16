#!/bin/bash
#SBATCH -J gruboros_test          # Job name
#SBATCH -o logs/test_%j.out       # Output file
#SBATCH -e logs/test_%j.err       # Error file
#SBATCH -p h100                   # H100 partition
#SBATCH -N 1                      # Single node test
#SBATCH --ntasks=1                # Single task to check GPU count
#SBATCH -t 00:10:00              # 10 minute test
#SBATCH -A <ALLOCATION>          # Your allocation

set -e

# Load modules
module load gcc/13.1.0
module load cuda/12.2
module load python3/3.11.8

# Activate environment
source gruboros_env/bin/activate  # Adjust based on your setup

# Check system configuration
echo "=== Node Information ==="
hostname
echo ""

echo "=== GPU Information ==="
nvidia-smi -L || echo "nvidia-smi not available"
echo ""

echo "=== CUDA Information ==="
nvcc --version || echo "nvcc not available"
echo ""

echo "=== Python GPU Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    SMs: {props.multi_processor_count}')
"

echo ""
echo "=== Environment Variables ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
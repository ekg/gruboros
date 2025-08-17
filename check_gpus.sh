#!/bin/bash
#SBATCH -J gpu_check
#SBATCH -o gpu_check_%j.out
#SBATCH -e gpu_check_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00

echo "Checking GPUs on $(hostname)"
echo "================================"

# Try nvidia-smi
echo "nvidia-smi -L output:"
nvidia-smi -L

echo ""
echo "nvidia-smi full output:"
nvidia-smi

echo ""
echo "Number of GPUs detected:"
nvidia-smi -L | wc -l

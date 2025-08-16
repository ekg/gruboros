#!/bin/bash
# setup_stampede3.sh - Setup script for gruboros on TACC Stampede3

set -e

echo "=== Setting up gruboros on Stampede3 ==="

# Load necessary modules for H100 nodes
echo "Loading required modules..."
module load gcc/13.1.0
module load cuda/12.2
module list

# Create micromamba environment
echo -e "\nCreating micromamba environment..."
if micromamba env list | grep -q "^gruboros "; then
    echo "Environment 'gruboros' already exists. Removing..."
    micromamba env remove -n gruboros -y
fi

# Use the clean environment file
micromamba env create -f environment_clean.yml -y

echo -e "\nActivating environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate gruboros

# Install pip packages that aren't in conda
echo -e "\nInstalling additional pip packages..."
pip install accelerate
pip install git+https://github.com/facebookresearch/schedule_free.git

# Verify installation
echo -e "\n=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo -e "\nChecking other dependencies..."
python -c "
import deepspeed
import accelerate
import schedulefree
import numpy
import pandas
import psutil
print('All core dependencies imported successfully!')
"

echo -e "\n=== Setup complete! ==="
echo "To activate the environment in future sessions, run:"
echo "  micromamba activate gruboros"
echo ""
echo "To run training, use:"
echo "  ./train.cuda.sh"
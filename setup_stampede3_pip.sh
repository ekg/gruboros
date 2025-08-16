#!/bin/bash
# setup_stampede3_pip.sh - Quick pip-only setup for Stampede3

set -e

echo "=== Quick pip setup for gruboros on Stampede3 ==="

# Load necessary modules
echo "Loading required modules..."
module load gcc/13.1.0
module load cuda/12.2
module load python3/3.11.8

# Create virtual environment
echo -e "\nCreating Python virtual environment..."
python3 -m venv gruboros_env

# Activate environment
source gruboros_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo -e "\nInstalling PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo -e "\nInstalling other dependencies..."
pip install numpy scipy psutil tqdm deepspeed accelerate
pip install git+https://github.com/facebookresearch/schedule_free.git

# Optional: pandas and pyarrow if needed
# pip install pandas pyarrow

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

echo -e "\n=== Setup complete! ==="
echo "To activate the environment in future sessions, run:"
echo "  source gruboros_env/bin/activate"
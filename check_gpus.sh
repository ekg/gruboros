#!/bin/bash
# Check GPU environment on Frontier

module load rocm/6.2.4

echo "===== GPU Information ====="
rocm-smi --showdriverversion
echo "===== GPU Cards ====="
rocm-smi
echo "===== ROCM Environment ====="
env | grep ROCM
echo "===== HIP Environment ====="
env | grep HIP
echo "===== NCCL Environment ====="
env | grep NCCL
echo "===== Python Environment ====="
which python
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
echo "===== Done ====="

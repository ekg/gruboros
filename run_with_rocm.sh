#!/bin/bash
# Wrapper script that sets up the right environment and patches DeepSpeed for ROCm

# Set ROCm environment variables
export CUDA_HOME=/opt/rocm-6.2.4
export HIP_CLANG_PATH=/opt/rocm-6.2.4/llvm
export ROCM_HOME=/opt/rocm-6.2.4
export DS_SKIP_CUDA_CHECK=1  # Skip DeepSpeed's CUDA check

# Directory for extension building
export TORCH_EXTENSIONS_DIR="$PWD/deepspeed_rocm_extensions"
mkdir -p $TORCH_EXTENSIONS_DIR

# Apply the patch
if [ ! -f ".deepspeed_patched" ]; then
  echo "Patching DeepSpeed for ROCm compatibility..."
  python deespeed_rocm_patch.py
  touch .deepspeed_patched
fi

# Run the deepspeed command with all arguments passed through
sbatch frontier_train.sh

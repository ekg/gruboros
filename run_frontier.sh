#!/bin/bash
# Simple wrapper script to launch minLM training on Frontier
# No complex setup, just environment variables that actually work

# Initialize logging
echo "========= minLM Training on Frontier ========="
echo "Date: $(date)"
echo "===== Environment Setup ====="

# Environment preparation
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.2.4
export ROCM_HOME=/opt/rocm-6.2.4

# This is the key for ROCm compatibility
export PYTORCH_ROCM_ARCH=gfx90a
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# Make training script executable
chmod +x srun_train.sh

# Launch the training script
./srun_train.sh

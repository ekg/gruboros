#!/bin/bash
# Pre-build script for DeepSpeed with ROCm on Frontier
# This helps initialize DeepSpeed kernels for AMD GPUs

# Load necessary modules
module load PrgEnv-amd
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Set ROCm environment variables
export CUDA_HOME=/opt/rocm-6.2.4
export HIP_CLANG_PATH=/opt/rocm-6.2.4/llvm
export ROCM_HOME=/opt/rocm-6.2.4

# DeepSpeed build options for ROCm
export DS_BUILD_OPS=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

# Path for build outputs
export TORCH_EXTENSIONS_DIR="$PWD/deepspeed_rocm_extensions"

# Create the directory if it doesn't exist
mkdir -p $TORCH_EXTENSIONS_DIR

# Print environment for debugging
echo "Building DeepSpeed ROCm extensions with:"
echo "ROCM_HOME: $ROCM_HOME"
echo "CUDA_HOME: $CUDA_HOME"
echo "HIP_CLANG_PATH: $HIP_CLANG_PATH"
echo "Extensions directory: $TORCH_EXTENSIONS_DIR"

# Pre-build commonly used ops to ensure they're ready for training
python -c "import deepspeed; deepspeed.ops.op_builder.FusedAdamBuilder().load()"
python -c "import deepspeed; deepspeed.ops.op_builder.UtilsBuilder().load()"

echo "DeepSpeed ROCm extensions pre-built successfully"

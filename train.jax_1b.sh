#!/bin/bash
set -e -x

# =============================================================================
# JAX GRU+EMA 1B Training - DDP mode with 8 GPUs
# =============================================================================
# Model: dim=2048, depth=24 → ~1.01B params
# Style: residuals, EMA (alpha=0.01), no expansion, no ff
# =============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=/home/erikg/micromamba/envs/mingru/bin:$PATH
# XLA_FLAGS removed - triton flags not valid for this JAX version

# Use nvidia pip packages for CUDA libs (cuDNN 9.16 compatible with JAX 0.8.1)
NVIDIA_LIBS="/home/erikg/micromamba/envs/mingru/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/cudnn/lib:${NVIDIA_LIBS}/cublas/lib:${NVIDIA_LIBS}/cusparse/lib:${NVIDIA_LIBS}/cufft/lib:${NVIDIA_LIBS}/curand/lib:${NVIDIA_LIBS}/cusolver/lib:${NVIDIA_LIBS}/nccl/lib:${NVIDIA_LIBS}/nvjitlink/lib:${LD_LIBRARY_PATH}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_jax_gru_ema_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "======================================================================="
echo "=== JAX GRU+EMA 1B Training ==="
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/python -u jax_gru_ema/train_1b.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  --dim 2048 \
  --depth 24 \
  --expansion 1.0 \
  --ff_mult 0.0 \
  --ema_alpha 0.01 \
  --vocab_size 50281 \
  \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  --batch_size 16 \
  --chunk_size 512 \
  \
  --lr 0.001 \
  --weight_decay 0.033 \
  --warmup_steps 100 \
  --train_steps 100000 \
  --grad_clip 1.0 \
  \
  --log_every 10 \
  \
  --ddp \
  2>&1 | tee "logs/jax_gru_ema_1b_${TIMESTAMP}.log"

echo "Done!"

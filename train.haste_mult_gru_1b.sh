#!/bin/bash
set -e -x

# =============================================================================
# HasteGRU_MultLM 1B Training - Fused CUDA kernel with native BF16
# =============================================================================
# Model: ~1B params using Haste's fused GRU_SiLU CUDA kernel
# Architecture:
#   - EXACT numerical equivalence to CuDNNGRU_MultLM with gate_activation='silu'
#   - Fused GRU + SiLU multiplicative gate in single CUDA kernel
#   - Native BF16 support (no fp32 conversion overhead)
#   - Same weight tying, LayerNorm, and residual structure
#
# Key differences from CuDNN version:
#   - Uses --use_haste_mult_gru instead of --use_cudnn_mult_gru
#   - Uses --no-autocast (haste kernel handles BF16 natively)
#   - No --compile (haste CUDA kernels are already optimized)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_haste_mult_gru_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29550

echo "======================================================================="
echo "=== HasteGRU_MultLM 1B - Fused CUDA kernel, BF16 native           ===="
echo "======================================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Config: dim=2048, depth=27, expansion=1.0"
echo "Batch: 6 x 2048 x 8 GPUs = 98,304 tokens/step"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================="

# HasteGRU_MultLM 1B config: dim=2048, depth=27, expansion=1.0
# Matches CuDNNGRU_MultLM settings exactly for fair comparison

/home/erikg/micromamba/envs/mingru/bin/torchrun \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  \
  --tokenizer tiktoken \
  --tiktoken_encoding p50k_base \
  \
  --dim 2048 \
  --depth 27 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_haste_mult_gru \
  --no-tbptt \
  \
  --chunk_size 2048 \
  --batch_size 6 \
  --grad_accum 1 \
  \
  --train_steps 10000 \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.995 \
  --weight_decay 0.033 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 5 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  --no-autocast \
  2>&1 | tee "logs/haste_mult_gru_1b_${TIMESTAMP}.log"

echo "Done!"

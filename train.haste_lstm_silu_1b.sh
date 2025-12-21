#!/bin/bash
set -e -x

# =============================================================================
# HasteLSTMSilu 1B Training - Fused CUDA kernel with native BF16
# =============================================================================
# Model: ~1B params using fused LSTM_SiLU CUDA kernel
# Architecture:
#   - Recurrence: fused LSTM + silu gate in single CUDA kernel
#   - LSTM has separate cell state (c) for memory, hidden state (h) for output
#   - LSTM gates: input (i), forget (f), output (o), cell (g)
#   - Native BF16 support (no fp32 conversion overhead)
#   - Fused selectivity: output = h * silu(Wg_x @ x + Wg_h @ h + bg)
#
# LSTM vs GRU:
#   - LSTM has separate cell state for long-range memory
#   - Potentially better at modeling long dependencies
#   - More parameters per layer (4 gates vs 3 for GRU)
#
# This uses the new haste fork (gh/ekg/haste) with LSTM_SiLU kernel.
# Settings matched to Mamba2 for fair comparison.
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_haste_lstm_silu_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29525

echo "======================================================================="
echo "=== HasteLSTMSilu 1B - Fused LSTM+SiLU CUDA kernel, BF16 native   ===="
echo "======================================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Config: dim=2048, depth=32, expansion=1.0"
echo "Batch: 24 x 512 x 8 GPUs = 98,304 tokens/step"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================="

# HasteLSTMSilu 1B config: dim=2048, depth=32, expansion=1.0
# LSTM has more params per layer (4 gates vs 3 for GRU)
# Matched to Mamba2 settings: lr=0.001, batch=24, chunk=512

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
  --depth 32 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_haste_lstm_silu \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 24 \
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
  2>&1 | tee "logs/haste_lstm_silu_1b_${TIMESTAMP}.log"

echo "Done!"

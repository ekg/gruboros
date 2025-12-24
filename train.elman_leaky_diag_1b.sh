#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeakyDiag 1B Training - DIAGONAL R (like Mamba's diagonal A!)
# =============================================================================
# Architecture:
#   - candidate = tanh(r ⊙ h + Wx @ x + b)          -- r is [D] not [D,D]!
#   - delta = sigmoid(W_delta @ x + b_delta)        -- input-dependent blend
#   - h_new = (1 - delta) * h + delta * candidate   -- leaky integration
#   - gate = silu(W_gate @ x + b)                   -- INPUT-ONLY output gate
#   - output = h_new * gate
#
# Benefits:
#   - D params instead of D² for recurrence (2048 vs 4M params!)
#   - No GEMM needed - pure elementwise ops (faster!)
#   - Cross-channel mixing via Wx @ x and depth
#   - Matches Mamba's diagonal A architecture
#
# Params: ~1.05B (depth=35, expansion=1.2) to match Mamba2's ~1.0B
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_leaky_diag_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29576

echo "======================================================================="
echo "=== ElmanLeakyDiag 1B (diagonal R, like Mamba's diagonal A!) ==="
echo "======================================================================="

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
  --depth 35 \
  --expansion_factor 1.2 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_leaky_diag \
  --delta_init -2.0 \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 24 \
  --grad_accum 1 \
  \
  --train_steps 3000 \
  --lr 0.0001 \
  --weight_decay 0.1 \
  --grad_clip 0.1 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/elman_leaky_diag_1b_${TIMESTAMP}.log"

echo "Done!"

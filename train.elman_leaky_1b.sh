#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeaky 1B Training - True Discretized Dynamics with Input-Dependent Delta
# =============================================================================
# Model: ~1B params using haste ElmanLeaky CUDA kernels
# Architecture:
#   - candidate = tanh(R @ h + Wx @ x + b)  -- R sees properly blended h!
#   - delta = sigmoid(W_delta @ x + b_delta)  -- input-dependent
#   - h_new = (1 - delta) * h + delta * candidate  -- leaky integration
#
# Key insight: This is the CORRECT discretization of continuous-time RNN
# dynamics. Unlike the broken "fast approximation", the recurrence matrix R
# sees the EMA-smoothed state h[t-1], not the raw Elman output.
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_leaky_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29570

echo "======================================================================="
echo "=== ElmanLeaky 1B (True Discretized Elman, Input-Dependent Delta) ==="
echo "======================================================================="

# ElmanLeaky 1B config: dim=2048, depth=32, ff_mult=0.0
# delta_init=-2.0 → sigmoid(-2)≈0.12 (slow dynamics initially)

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
  --use_elman_leaky \
  --delta_init -2.0 \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 3000 \
  --lr 0.0006 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 500 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/elman_leaky_1b_${TIMESTAMP}.log"

echo "Done!"

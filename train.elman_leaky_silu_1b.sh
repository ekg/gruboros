#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeakySilu 1B Training - silu activation + leaky integration
# =============================================================================
# Architecture:
#   - candidate = silu(R @ h + Wx @ x + b)          -- silu instead of tanh!
#   - delta = sigmoid(W_delta @ x + b_delta)        -- input-dependent blend
#   - h_new = (1 - delta) * h + delta * candidate   -- leaky integration
#   - gate = silu(W_gate @ x + b)                   -- INPUT-ONLY output gate
#   - output = h_new * gate
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_leaky_silu_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29575

echo "======================================================================="
echo "=== ElmanLeakySilu 1B (silu + leaky integration) ==="
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
  --depth 32 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_leaky_silu \
  --delta_init -2.0 \
  --recurrence_chunk_size 64 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
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
  2>&1 | tee "logs/elman_leaky_silu_1b_${TIMESTAMP}.log"

echo "Done!"

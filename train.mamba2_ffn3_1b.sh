#!/bin/bash
set -e -x

# =============================================================================
# Mamba2 + 3-layer FFN Training - DDP mode with 8 GPUs
# =============================================================================
# Based on arXiv:2505.06633: "Attention Is Not All You Need"
# Key finding: 3-layer FFN (d→4d→4d→d) with fewer blocks outperforms
# standard 2-layer FFN (d→4d→d) with more blocks at same param count.
#
# Model: ~1.1B params with Mamba2 SSD + 3-layer FFN with GELU
# Structure per block:
#   h = x + Mamba2(norm(x))      # Linear SSM for sequence mixing
#   out = h + FFN3(norm(h))      # 3-layer FFN for nonlinear transformations
#
# FFN3 adds TWO GELU nonlinearities per block (vs one in standard FFN)
# This may help with the "injectivity" issue that linear SSMs have.
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mamba2_ffn3_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29523

echo "======================================================================="
echo "=== Mamba2 + 3-layer FFN 1.1B Training ==="
echo "======================================================================="

# Mamba2+FFN3 1.1B config: dim=2048, depth=12
# depth=12 gives ~1.1B params (fewer blocks needed due to 3-layer FFN)
# Per block: Mamba2 ~25M + FFN3 ~67M (3x 2048x8192 matrices)

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
  --depth 15 \
  --dropout 0.0 \
  \
  --use_mamba2_ffn3 \
  --mamba_d_state 64 \
  --mamba_expand 2 \
  \
  --chunk_size 512 \
  --batch_size 20 \
  --grad_accum 1 \
  \
  --train_steps 10000 \
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
  2>&1 | tee "logs/mamba2_ffn3_1b_${TIMESTAMP}.log"

echo "Done!"

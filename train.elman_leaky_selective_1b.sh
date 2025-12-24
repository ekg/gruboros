#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeakySelective 1B Training - Mamba2-style Discretization + h+x Output Gate
# =============================================================================
# Model: ~1B params using haste ElmanLeakySelective CUDA kernels
# Architecture:
#   - candidate = tanh(R @ h + Wx @ x + b)       -- NONLINEAR (our innovation!)
#   - dt = softplus(W_delta @ x + b_delta)       -- input-dependent timestep
#   - alpha = exp(-dt * exp(A))                  -- Mamba2-style per-channel decay
#   - h_new = alpha * h + (1 - alpha) * candidate -- exponential blend
#   - gate = silu(W_gate_x @ x + W_gate_h @ h)   -- h+x selective output
#   - output = h * gate
#
# Key features:
# - Per-channel decay rates A (like Mamba2's diagonal A matrix)
# - Input-dependent timestep via softplus (like Mamba2's Δ)
# - NONLINEAR candidate (tanh) - our innovation beyond Mamba2!
# - h+x output gate for richer selectivity (both x AND h!)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_leaky_selective_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29571

echo "======================================================================="
echo "=== ElmanLeakySelective 1B (Mamba2-style + h+x Output Gate) ==="
echo "======================================================================="

# ElmanLeakySelective 1B config: dim=2048, depth=32, ff_mult=0.0
# delta_init=-2.0 → softplus(-2)≈0.13 (slow dynamics initially)

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
  --use_elman_leaky_selective \
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
  2>&1 | tee "logs/elman_leaky_selective_1b_${TIMESTAMP}.log"

echo "Done!"

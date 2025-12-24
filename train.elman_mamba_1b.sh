#!/bin/bash
set -e -x

# =============================================================================
# ElmanMamba 1B Training - CLEANER Mamba2-style (INPUT-ONLY output gate)
# =============================================================================
# Model: ~1B params using haste ElmanMamba CUDA kernels
# Architecture:
#   - candidate = tanh(R @ h + Wx @ x + b)       -- NONLINEAR (our innovation!)
#   - dt = softplus(W_delta @ x + b_delta)       -- input-dependent timestep
#   - decay_rate = exp(-exp(A_log))              -- LOG-SPACE: always in (0,1)!
#   - alpha = exp(-dt * decay_rate)              -- Mamba2-style per-channel decay
#   - h_new = alpha * h + (1 - alpha) * candidate -- exponential blend
#   - gate = silu(W_gate @ x + b)                -- INPUT-ONLY output gate!
#   - output = h_new * gate
#
# Key differences from ElmanLeakySelective:
# - OUTPUT GATE depends ONLY on x, NOT h! (like Mamba2's C matrix)
# - Simpler, closer to actual Mamba2 architecture
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_mamba_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29573

echo "======================================================================="
echo "=== ElmanMamba 1B (Mamba2-style + INPUT-ONLY Output Gate) ==="
echo "======================================================================="

# ElmanMamba 1B config: dim=2048, depth=32, ff_mult=0.0
# delta_init=3.0 → dt=softplus(3)≈3.0
# A in (-0.5, 0.5) → decay_rate=exp(-exp(A))≈0.37 → alpha=exp(-1.1)≈0.33 → ~67% candidate

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
  --use_elman_mamba \
  --delta_init 3.0 \
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
  2>&1 | tee "logs/elman_mamba_1b_${TIMESTAMP}.log"

echo "Done!"

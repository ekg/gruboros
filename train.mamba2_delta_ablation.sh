#!/bin/bash
set -e -x

# =============================================================================
# Mamba2-Style Delta Ablation - Testing softplus/exp delta vs sigmoid
# =============================================================================
# Our baseline ElmanLeakyCompeteSilu uses sigmoid for delta:
#   delta = sigmoid(W_delta @ x)
#   h_new = (1-delta)*h + delta*candidate
#
# Mamba2 uses softplus/exp (log-space parameterization):
#   delta_raw = W_delta @ x  (can be any value)
#   delta = softplus(delta_raw)  (always positive, better gradients)
#   decay = exp(-delta)  (between 0 and 1)
#   h_new = decay*h + (1-decay)*candidate
#
# Benefits of Mamba2-style:
# - Wider dynamic range (delta can be 0.001 to 100+)
# - Better gradients (softplus doesn't saturate like sigmoid)
# - More natural for log-space analysis
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mamba2_delta"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29575

echo "======================================================================="
echo "=== Mamba2-Style Delta (softplus/exp instead of sigmoid) ==="
echo "======================================================================="
echo "Architecture: R@h recurrence + tanh candidate + Mamba2 delta + compete×silu"
echo "This tests whether log-space delta (softplus/exp) improves over sigmoid"
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
  --use_elman_leaky_mamba2_delta \
  --compete_n_groups 32 \
  --recurrence_chunk_size 64 \
  --delta_init -1.8 \
  --no-tbptt \
  \
  --chunk_size 512 \
  --batch_size 16 \
  --grad_accum 1 \
  \
  --train_steps 300 \
  --lr 0.0006 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  \
  --save_every 100 \
  --keep_checkpoints 3 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16 \
  2>&1 | tee "logs/mamba2_delta_${TIMESTAMP}.log"

echo "Done!"

#!/bin/bash
set -e -x

# =============================================================================
# EMA GRU TEST - Long-range memory via EMA path
# =============================================================================
# Same working config as minLM test, but with EMA path for long-range memory
# EMA alpha=0.01 gives ~70 token half-life (vs GRU's ~1-2 tokens)
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_ema_gru_test"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

echo "======================================================================="
echo "=== EMA GRU TEST - Long-range memory ==="
echo "======================================================================="
echo "Architecture: HybridFusedGRU_EMA (GRU + parallel EMA path)"
echo "EMA alpha=0.01 (half-life ~70 tokens)"
echo "z_bias=0.0 (neutral gates)"
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
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
  --depth 20 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_ema_gru \
  --ema_alpha 0.01 \
  --z_bias_input 0.0 \
  --z_bias_hidden 0.0 \
  \
  --chunk_size 512 \
  --batch_size 32 \
  --grad_accum 8 \
  \
  --train_steps 5000 \
  --lr 0.001 \
  --sgd \
  --momentum 0.0 \
  --weight_decay 0.033 \
  --grad_clip 0.0 \
  \
  --save_every 1000 \
  --keep_checkpoints 10 \
  --keep_elite 32 \
  --milestone_every 10000 \
  \
  --ddp \
  --ddp-find-unused \
  --cuda \
  --bf16

echo "Done!"

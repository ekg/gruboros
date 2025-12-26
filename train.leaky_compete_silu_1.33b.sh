#!/bin/bash
set -e -x

# =============================================================================
# ElmanLeakyCompeteSilu ~1.33B params - matched to Mamba2 for fair comparison
# =============================================================================
# depth=38 gives 1.322B params (close to Mamba2's 1.33B)
# Uses compete×silu gate with leaky integration
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_leaky_compete_silu_1.33b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

echo "======================================================================="
echo "=== ElmanLeakyCompeteSilu (depth=38, ~1.33B params) ==="
echo "======================================================================="
echo "Compete×silu gate with leaky integration, matched to Mamba2 params"
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
  --depth 38 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --dropout 0.0 \
  \
  --use_elman_leaky_compete_silu \
  --compete_n_groups 32 \
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
  2>&1 | tee "logs/leaky_compete_silu_1.33b_${TIMESTAMP}.log"

echo "Done!"

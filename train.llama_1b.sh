#!/bin/bash
set -e -x

# =============================================================================
# LLaMA-style Transformer Baseline - ~1.3B params
# =============================================================================
# Modern transformer (RoPE, SwiGLU, RMSNorm) for comparison with:
# - Triple R (1.28B): 5.179 avg50
# - Mamba2 (1.33B): running
# - Baseline ElmanLeakyCompeteSilu (1.15B): 5.223 avg50
#
# LLaMA 1.3B: dim=2048, depth=24, n_heads=32 = 1.34B params
# =============================================================================

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_llama_1b"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29720

echo "======================================================================="
echo "=== LLaMA Transformer 1.3B Baseline ==="
echo "======================================================================="
echo "dim=2048, depth=24, n_heads=32, ff_mult=4.0 (SwiGLU)"
echo "RoPE position embeddings, RMSNorm"
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
  --depth 24 \
  --ff_mult 4.0 \
  --dropout 0.0 \
  \
  --use_llama \
  --transformer_n_heads 32 \
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
  2>&1 | tee "logs/llama_1b_${TIMESTAMP}.log"

echo "Done!"

#!/bin/bash
# FlashRNN GRU - ~500M parameters (stable configuration)
# Uses multi-head configuration: 32 heads × 64 head_dim = 2048
# Stock upstream FlashRNN with 'gru' function, 'cuda' backend
# Reduced depth to 20 for numerical stability

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_flashrnn_stable"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29535
# CRITICAL: ninja must be in PATH for FlashRNN JIT compilation
export PATH="/home/erikg/micromamba/envs/mingru/bin:$PATH"

echo "================================================================"
echo "  FlashRNN GRU - Stable Configuration"
echo "================================================================"
echo "  Architecture: 32 heads × 64 head_dim = 2048"
echo "  dim=2048, depth=20, expansion=1.0"
echo "  chunk_size=512, batch_size=16"
echo "  Stock upstream FlashRNN"
echo "================================================================"

/home/erikg/micromamba/envs/mingru/bin/torchrun \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --data /mnt/nvme2n1/erikg/pile.txt \
    --output "$OUTPUT_DIR" \
    --tokenizer tiktoken \
    --tiktoken_encoding p50k_base \
    --dim 2048 \
    --depth 20 \
    --expansion_factor 1.0 \
    --ff_mult 0.0 \
    --dropout 0.0 \
    --use_flash_gru \
    --chunk_size 512 \
    --batch_size 16 \
    --grad_accum 4 \
    --train_steps 500 \
    --lr 0.0003 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 500 \
    --keep_checkpoints 10 \
    --milestone_every 2000 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "logs/flashrnn_stable_${TIMESTAMP}.log"

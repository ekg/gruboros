#!/bin/bash
# FlashRNN GRU - 2048 context FAST training
# FlashRNN provides 2-5x speedup over cuDNN via fused CUDA kernels
# Should achieve >20k T/s per GPU (vs ~5k with cuDNN checkpointing)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_flashrnn_2048"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29515
# CRITICAL: ninja must be in PATH for FlashRNN JIT compilation
export PATH="/home/erikg/micromamba/envs/mingru/bin:$PATH"
# Verify ninja is available
echo "ninja version: $(/home/erikg/micromamba/envs/mingru/bin/ninja --version)"

echo "================================================================"
echo "  FlashRNN GRU - 2048 Context OPTIMIZED"
echo "================================================================"
echo "  Using FlashRNN fused CUDA kernels (2-5x over cuDNN!)"
echo "  dim=2048, depth=27, expansion=1.0, ~1B params"
echo "  chunk_size=2048, batch_size=64 (using freed VRAM!)"
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
    --depth 27 \
    --expansion_factor 1.0 \
    --ff_mult 0.0 \
    --dropout 0.0 \
    --use_flash_gru \
    --chunk_size 2048 \
    --batch_size 64 \
    --grad_accum 1 \
    --train_steps 100 \
    --lr 0.001 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "logs/flashrnn_2048_${TIMESTAMP}.log"

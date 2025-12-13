#!/bin/bash
# cuDNN GRU - 1B parameters (stable baseline)
# StandardGRU uses PyTorch nn.GRU with cuDNN backend - proven stable
#
# Architecture: dim=2048, depth=24, expansion=1.0
# Parameters: ~1.01B
# This is the gold standard nonlinear GRU implementation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_gru_1b"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29540

echo "================================================================"
echo "  cuDNN GRU - 1B Parameter Baseline"
echo "================================================================"
echo "  Architecture: StandardGRU (cuDNN-optimized)"
echo "  dim=2048, depth=24, expansion=1.0, ~1.01B params"
echo "  chunk_size=512, batch_size=16, grad_accum=4"
echo "  8 GPUs with DDP"
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
    --depth 24 \
    --expansion_factor 1.0 \
    --ff_mult 0.0 \
    --dropout 0.0 \
    --use_standard_gru \
    --recurrence_chunk_size 64 \
    --chunk_size 512 \
    --batch_size 16 \
    --grad_accum 4 \
    --train_steps 10000 \
    --lr 0.001 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 1000 \
    --keep_checkpoints 10 \
    --milestone_every 5000 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "logs/cudnn_gru_1b_${TIMESTAMP}.log"

echo "Training finished."

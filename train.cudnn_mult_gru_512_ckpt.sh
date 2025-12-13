#!/bin/bash
# cuDNN GRU + Multiplicative Gating 1B training with 512 chunk size
# WITH gradient checkpointing (128 token inner chunks) for validation
# This should match the non-checkpointed version exactly

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_mult_gru_512_ckpt"
LOG_FILE="logs/cudnn_mult_gru_512_ckpt_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512

echo "Starting cuDNN Mult GRU 512 chunk WITH CHECKPOINTING at $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Checkpointing: inner_chunk_size=128"

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
    --use_cudnn_mult_gru \
    --use_checkpointing \
    --inner_chunk_size 128 \
    --chunk_size 512 \
    --batch_size 24 \
    --grad_accum 1 \
    --train_steps 10000 \
    --lr 0.001 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 2500 \
    --keep_checkpoints 10 \
    --keep_elite 32 \
    --milestone_every 5000 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "$LOG_FILE"

echo "Training completed at $(date)"

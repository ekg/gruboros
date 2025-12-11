#!/bin/bash
# cuDNN GRU + 3-layer FFN Gate 1B training
# Tests deep selectivity hypothesis: 3-layer FFN on concat[x,h] for gate
# Gate: sigmoid(FFN3(concat(x, h))) - both input and hidden matter

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_ffn3_gru_1b"
LOG_FILE="logs/cudnn_ffn3_gru_1b_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29511

echo "Starting cuDNN GRU + FFN3 Gate 1B training at $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

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
    --depth 14 \
    --expansion_factor 1.0 \
    --ffn_expand 2.0 \
    --ff_mult 0.0 \
    --dropout 0.0 \
    --use_cudnn_ffn3_gru \
    --chunk_size 256 \
    --batch_size 48 \
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

#!/bin/bash
# Production 1B SequentialTritonGRU - FIXED architecture + memory
# Config: dim=2048, depth=27, batch=24, chunk=512 (matches cuDNN)
# Commits: 91435a5 (architecture fix), 33106ec (memory fix)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_1b_sequential_512_33106ec"
LOG_FILE="logs/1b_sequential_512_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29530

echo "========================================================================"
echo "Starting 1B SequentialTritonGRU Training - FIXED"
echo "========================================================================"
echo "Timestamp: $(date)"
echo "Config: dim=2048, depth=27, batch=24, chunk=512, bf16"
echo "Architecture: NOW MATCHES StandardGRU (added input_proj layer)"
echo "Memory: FIXED (preallocate output tensor, no clone spam)"
echo "Verified: Loss 10.82 → 7.67 in 10 steps (smooth learning!)"
echo "Commits: 91435a5 (arch fix), 33106ec (memory fix)"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "========================================================================"

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
    --use_sequential_triton_gru \
    --chunk_size 512 \
    --batch_size 24 \
    --grad_accum 1 \
    --train_steps 10000 \
    --lr 0.001 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 500 \
    --keep_checkpoints 10 \
    --keep_elite 32 \
    --milestone_every 1000 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Training completed at $(date)"
echo "========================================================================"

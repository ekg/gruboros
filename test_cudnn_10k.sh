#!/bin/bash
# 10K step test of StandardGRU (cuDNN) for performance comparison
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29541

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/test_cudnn_10k_${TIMESTAMP}"
LOG_FILE="logs/test_cudnn_10k_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "StandardGRU (cuDNN) 10K Step Test - Performance Baseline"
echo "========================================================================"
echo "Timestamp: $(date)"
echo "Config: 1.11B params (dim=2048, depth=27, batch=24, chunk=512)"
echo "Architecture: StandardGRU with cuDNN backend"
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
    --dim 2048 --depth 27 --expansion_factor 1.0 --ff_mult 0.0 --dropout 0.0 \
    --use_standard_gru --chunk_size 512 --batch_size 24 --grad_accum 1 \
    --train_steps 10000 --lr 0.001 --weight_decay 0.033 --grad_clip 1.0 \
    --save_every 1000 --keep_checkpoints 10 --keep_elite 10 \
    --milestone_every 2500 --ddp --ddp-find-unused --cuda --bf16 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Test completed at $(date)"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"
echo "========================================================================"

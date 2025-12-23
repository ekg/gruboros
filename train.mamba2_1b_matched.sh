#!/bin/bash
# Mamba2 with IDENTICAL config for fair comparison with HasteGRU_MultLM
#
# Matched to train.haste_mult_gru_1b.sh:
# - batch_size=24, chunk_size=512 (same batch shape)
# - Same optimizer: lr=0.001, sf_beta=0.9/0.995, weight_decay=0.033
# - Same 10k steps
# - depth=35 gives ~1.0B params (vs haste depth=27 ~1.0B)
set -e -x

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_mamba2_1b_matched"
LOG_FILE="logs/mamba2_1b_matched_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29551

echo "========================================================================"
echo "Mamba2 SSM 1B - MATCHED config to HasteGRU_MultLM"
echo "========================================================================"
echo "Timestamp: $(date)"
echo "Config: ~1.0B params (dim=2048, depth=35, d_state=64, expand=2)"
echo "Batch: 24 x 512 x 8 GPUs = 98,304 tokens/step"
echo "Optimizer: Schedule-Free AdamW (same as HasteGRU)"
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
    --depth 35 \
    --dropout 0.0 \
    --use_mamba2 \
    --mamba_d_state 64 \
    --mamba_expand 2 \
    --chunk_size 512 \
    --batch_size 24 \
    --grad_accum 1 \
    --train_steps 3000 \
    --lr 0.001 \
    --sf_beta 0.9 \
    --sf_beta2 0.995 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --save_every 500 \
    --keep_checkpoints 5 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Training completed at $(date)"
echo "Log: $LOG_FILE"
echo "Compare with: logs/haste_mult_gru_1b_*.log"
echo "========================================================================"

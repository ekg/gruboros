#!/bin/bash
# Mult GRU with SwiGLU-style output (like LLaMA/Mistral FFN)
# Reduced batch size (20 vs 24) to fit in 48GB GPUs
# SwiGLU formula: y = value * SiLU(gate + x)
set -e -x

ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/cudnn_mult_gru_swiglu_${TIMESTAMP}"
LOG_FILE="logs/cudnn_mult_gru_swiglu_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29549
export TORCH_DISTRIBUTED_TIMEOUT=3600s
export TORCH_DISTRIBUTED_BACKEND="nccl"

echo "========================================================================"
echo "Mult GRU with SwiGLU-style output (LLaMA/Mistral-style)"
echo "========================================================================"
echo "Timestamp: $(date)"
echo "Config: ~1.0B params (dim=2048, depth=27, batch=20, chunk=512)"
echo "Output: y = value * SiLU(gate_logits + x + b)"
echo "Batch reduced from 24->20 to fit 48GB GPUs"
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
    --use_cudnn_mult_gru \
    --use_swiglu \
    --chunk_size 512 \
    --batch_size 20 \
    --grad_accum 1 \
    --train_steps 10000 \
    --lr 0.001 \
    --sf_beta 0.9 \
    --sf_beta2 0.995 \
    --weight_decay 0.033 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --save_every 1000 \
    --keep_checkpoints 10 \
    --keep_elite 10 \
    --milestone_every 2500 \
    --gossip_mixing_rate 0.0 \
    --ddp \
    --ddp-find-unused \
    --cuda \
    --bf16 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Training completed at $(date)"
echo "Log: $LOG_FILE"
echo "Compare with: logs/cudnn_mult_gru_silu_*.log (standard SiLU gate)"
echo "========================================================================"

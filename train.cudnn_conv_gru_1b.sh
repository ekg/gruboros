#!/bin/bash
# Conv1d + cuDNN GRU 1B training
# Inspired by Mamba2: 4-wide causal conv1d before GRU for local context
#
# Architecture:
# x_conv = CausalConv1d(x, kernel_size=4)  # Local context (like Mamba2)
# h = cuDNN_GRU(x_conv)                     # Sequence mixing
# gate = sigma(W_x @ x + W_h @ h + b)       # Content+state-dependent gate
# h' = h * gate                             # Multiplicative interaction
# out = W_out @ h'                          # Output projection
#
# Key insight from Mamba2:
# - Mamba2 uses 4-wide causal conv1d BEFORE SSM for local context
# - Our ablation showed input-dependent gating (3.06) beat h-dependent (3.16)
# - Adding conv1d gives the GRU richer local input features

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_cudnn_conv_gru_1b"
LOG_FILE="logs/cudnn_conv_gru_1b_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29516

echo "Starting Conv1d + cuDNN GRU 1B training at $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Architecture: 4-wide causal conv1d -> GRU -> multiplicative gate"

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
    --use_cudnn_conv_gru \
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

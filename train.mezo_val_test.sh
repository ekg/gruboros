#!/bin/bash
set -e -x

# Quick test: validation after 10 steps to verify the fix

ulimit -n 65536

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30408
NUM_GPUS=8

export PYTORCH_DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== VALIDATION FIX TEST ==="
echo "Running 15 steps with validation at step 10"
echo "Should trigger validation and save checkpoint"

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "/tmp/mezo_val_test" \
  --params 500m \
  \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  \
  --use_causal_conv_gru \
  \
  --zero_order \
  --zo_method mezo \
  --zo_epsilon 0.0001 \
  --zo_num_perturbations_mezo 32 \
  \
  --chunk_size 2048 \
  --batch_size 16 \
  --grad_accum 4 \
  \
  --train_steps 15 \
  --lr 0.0003 \
  --sf_beta 0.9 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  --save_every 10 \
  --keep_checkpoints 1 \
  --validation_interval 10 \
  --validation_batches 32 \
  \
  --ddp \
  --cuda \
  --bf16

echo ""
echo "=== TEST COMPLETE ==="
ls -lh /tmp/mezo_val_test/checkpoints/ || echo "No checkpoints"

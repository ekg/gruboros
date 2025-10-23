#!/bin/bash
set -e -x

# === 8-Hour Pile Training Run ===
# Goal: Basic competence on copying/completion tasks
# Target: ~1B tokens, loss ~3.5-4.0

# --- Setup ---
ulimit -n 65536

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="500m"
NAME="${PARAMS}_pile_8hr"

GIT_HASH=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")
fi

if [ -n "$GIT_HASH" ]; then
    OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_${NAME}_${GIT_HASH}"
else
    OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_${NAME}"
fi

DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

JOB_ID=$(date +%s)
GOSSIP_TEMP_DIR="/tmp/gossip_temp_${JOB_ID}"
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/gossip"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${GOSSIP_TEMP_DIR}"

# --- Distributed Settings ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_TIMEOUT=3600s
export TORCH_DISTRIBUTED_BACKEND="gloo"
NUM_GPUS=8

# --- torch.compile Caching ---
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_DIR}/.inductor_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

echo "=== 8-Hour Pile Training ==="
echo "Goal: ~1B tokens, basic copying/completion competence"
echo "Model: 500M StandardGRU, TikToken vocab"
echo "Throughput target: ~35K tokens/sec (batch=8, chunk=2048, 8 GPUs)"
echo ""

# Calculate steps for 8 hours
# tokens/step = 8 GPUs × 8 batch × 2048 chunk = 131,072
# Target: 1B tokens = ~7,600 steps
# Time: 8 hours = 28,800 seconds
# seconds/step = 28,800 / 7,600 = 3.8s (reasonable with backprop)

TRAIN_STEPS=7600
VALIDATION_INTERVAL=500
SAVE_EVERY=1000

torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params $PARAMS \
  \
  `# TOKENIZATION` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  \
  `# ARCHITECTURE (NO FFN for speed)` \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.1 \
  \
  `# GRU (cuDNN optimized)` \
  --use_standard_gru \
  \
  `# SEQUENCES (maximize throughput)` \
  --chunk_size 2048 \
  --batch_size 8 \
  --grad_accum 1 \
  \
  `# TRAINING (8 hours to 1B tokens)` \
  --train_steps $TRAIN_STEPS \
  --lr 0.001 \
  --sf_beta 0.9 \
  --sf_beta2 0.999 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  \
  `# CHECKPOINTING (save progress)` \
  --save_every $SAVE_EVERY \
  --keep_checkpoints 10 \
  --keep_elite 50 \
  --archive_rate 0.01 \
  --validation_interval $VALIDATION_INTERVAL \
  --validation_batches 100 \
  \
  `# GOSSIP (evolutionary training)` \
  --gossip_merge_method recombination \
  --gossip_recombination_alpha 0.2 \
  --gossip_optimizer_recombination interpolate \
  --gossip_mixing_rate 0.001 \
  --gossip_p_value_threshold 0.05 \
  --gossip_lock_timeout 5.0 \
  --gossip_temp_dir "$GOSSIP_TEMP_DIR" \
  --gossip_fitness_window 5000 \
  --filesystem-coordinator \
  --fitness-weighted-checkpointing \
  --elite-checkpoint-multiplier 20.0 \
  \
  `# FLAGS` \
  --ddp \
  --cuda

echo ""
echo "=== Training Complete ==="
echo "Output directory: $OUTPUT_DIR"
echo "Expected performance:"
echo "  - Total tokens: ~1.0B"
echo "  - Final loss: ~3.5-4.0 (random: ~5.5)"
echo "  - Basic copying: Should work"
echo "  - Short completions: Partially coherent"
echo ""
echo "Next steps:"
echo "  1. Check best checkpoint in $OUTPUT_DIR"
echo "  2. Test with: python generate.py --model <checkpoint> --prompt 'The quick brown'"
echo "  3. If loss > 4.0, continue training another 4-8 hours"

rm -rf "$GOSSIP_TEMP_DIR"

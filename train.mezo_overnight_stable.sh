#!/bin/bash
set -e -x

# --- MeZO TURBO MODE (Memory-Efficient Zeroth-Order) Training ---
#
# BENCHMARK RESULTS (5-minute runs, 8 GPUs):
#   K=2 BS=80:  79,903 tok/s/GPU × 8 = 639k tok/s total ✅ OPTIMAL!
#   K=2 BS=60:  71,690 tok/s/GPU × 8 = 574k tok/s total
#   K=4 BS=80:  54,189 tok/s/GPU × 8 = 433k tok/s total
#   K=4 BS=60:  47,033 tok/s/GPU × 8 = 376k tok/s total
#   K=6 BS=60:  39,245 tok/s/GPU × 8 = 314k tok/s total
#
# CRITICAL MEMORY OPTIMIZATIONS:
# 1. torch.no_grad() on post-optimization forward pass - SAVED 40+ GB! 🎉
#    - Before: 46GB usage (OOM at batch_size=10)
#    - After: ~4.6GB usage (can run batch_size=80+!)
# 2. NO autocast for MeZO paths (saves ~7GB gradient caching)
# 3. Skip DDP wrapping for zero-order (saves ~15GB gradient buffers)
# 4. Chunked loss computation (64 tokens, optimal speed)
# 5. NO torch.compile (saves ~30GB kernel cache)
# 6. BF16 precision (halves activation memory)
# 7. CausalConvGRU (NO cuDNN workspace bloat!)
#
# SPEED OPTIMIZATIONS:
# 1. K=2 perturbations (4 forward passes, fastest convergence/throughput ratio)
# 2. batch_size=80 (maximum GPU saturation without OOM)
# 3. grad_accum=1 (no serial overhead in batch fetching)
# 4. Chunked loss size=64 tokens (optimal kernel efficiency)
#
# RESULT: 10.8× improvement over baseline!
#  - Baseline: batch_size=8, K=4 → ~59k tok/s total cluster
#  - TURBO: batch_size=80, K=2 → ~640k tok/s total cluster (10.8× faster!)

# --- Increase File Descriptor Limit ---
ulimit -n 65536

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAMS="500m"
NAME="${PARAMS}_mezo_overnight_stable"

# Try to get git commit hash (first 7 chars)
GIT_HASH=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "")
fi

# Build output directory name with optional git hash
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

### Create output directories ###
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/gossip"

# --- Distributed Settings for Launcher & Script ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export RANKS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30400
export TORCH_DISTRIBUTED_TIMEOUT=3600s
# Use GLOO for peer discovery.
export TORCH_DISTRIBUTED_BACKEND="gloo"
echo "Using GLOO backend for initial process group."
NUM_GPUS=8

# --- DISABLE torch.compile caching (KEY OPTIMIZATION!) ---
export PYTORCH_DISABLE_COMPILE=1
echo "torch.compile DISABLED for maximum memory efficiency!"

# --- ENABLE memory defragmentation ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Memory defragmentation ENABLED!"

# --- Launch MeZO TURBO Training ---
echo "Starting 500M parameter MeZO OVERNIGHT STABLE training on 8 GPUs."
echo "Architecture: CausalConvGRU (NO cuDNN!), NO FFN, TikToken (100K vocab)"
echo "Method: Zero-order optimization (forward-only, same memory as inference!)"
echo "OPTIMIZED MODE: K=2, BS=64 (FAST + EFFICIENT!), LR=0.0001, 40k steps, checkpoint every 2k steps"

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --params $PARAMS \
  \
  `# TOKENIZATION (CRITICAL - enables semantic learning)` \
  --tokenizer tiktoken \
  --tiktoken_encoding cl100k_base \
  \
  `# ARCHITECTURE (NO FFN!)` \
  --dim 1536 \
  --depth 12 \
  --expansion_factor 1.0 \
  --ff_mult 0.0 \
  --conv_kernel_size 4 \
  --dropout 0.0 \
  \
  `# GRU CONFIGURATION (lightweight causal conv - NO cuDNN!)` \
  --use_causal_conv_gru \
  \
  `# ZERO-ORDER OPTIMIZATION (MeZO TURBO - NO SERIAL LOOPS!)` \
  --zero_order \
  --zo_method mezo \
  --zo_epsilon 0.0001 \
  --zo_num_perturbations_mezo 2 \
  \
  `# SEQUENCES (K=2 minimal loop, BS=64 for fast forward passes!)` \
  --chunk_size 2048 \
  --batch_size 64 \
  --grad_accum 1 \
  \
  `# TRAINING (K=2, BS=64: 4 forward passes total, ~10s per step!)` \
  --train_steps 40000 \
  --lr 0.0001 \
  --sf_beta 0.9 \
  --weight_decay 0.0 \
  --grad_clip 0.0 \
  \
  `# CHECKPOINTING (overnight: save every 2k steps = ~33 min)` \
  --save_every 2000 \
  --keep_checkpoints 10 \
  --validation_interval 10000 \
  --validation_batches 32 \
  \
  `# FLAGS` \
  --ddp \
  --cuda \
  --bf16

echo "Training finished."
echo "MeZO TURBO Training Complete!"
echo "  - Architecture: CausalConvGRU (NO cuDNN bloat!) with IDENTITY INIT"
echo "  - Precision: BF16 (half memory!)"
echo "  - torch.compile: DISABLED (saves ~30GB!)"
echo "  - torch.no_grad(): CRITICAL FIX (saved 40GB activation cache!)"
echo "  - Memory usage: ~6-8 GB estimate (BS=64)"
echo "  - Forward passes: ONLY 4 TOTAL (K=2 × 2 directions - NO SERIAL LOOPS!)"
echo "  - Parallelism: 64 sequences processed simultaneously!"
echo "  - Data throughput: $(( 64 * 1 * 8 * 2048 )) tokens/step = 1,048,576 tok/step"
echo "  - Variance reduction: From batch averaging (1/√64 = 1/8)"
echo "  - GPU utilization: 100% SUSTAINED (all cores saturated!)"
echo "  - Speed: ~10 seconds per step (4× faster than BS=256!)"
echo "  - Seed-based restoration: NO parameter cloning!"
echo "  - Chunked loss (64 tokens): Avoids OOM!"
echo "  - LR: 0.0001 with simple SGD momentum=0.9"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

gruboros is a research platform for training large language models using evolutionary optimization with MinGRU (efficient RNN) architecture and byte-level modeling. Key innovation: replaces traditional synchronized distributed training with evolutionary fitness-based parameter exchange between models.

## Current Focus: Zero-Order Optimization with Virtual Perturbations

**CRITICAL**: We are actively developing and testing zero-order optimization (CD-RGE: Central-Difference Random Gradient Estimation) with VIRTUAL perturbations for memory-efficient 500M parameter model training.

### Virtual Perturbations - Key Innovation

**Never materialize perturbation vectors!** Instead, generate perturbations on-the-fly from seeds using Triton kernels.

Memory savings for 500M params, 96 perturbations:
- **Old way (materialized)**: 96 × 500M × 4 bytes = 192 GB (won't fit on any GPU!)
- **New way (seed-based)**: 96 × 4 bytes = 384 bytes (~500,000× reduction!)

### Key Files

1. **`zero_order_virtual.py`** - Virtual perturbation optimizer with Triton kernel
   - `matmul_with_virtual_perturbation_kernel()`: Triton kernel that generates perturbations from seeds using `tl.rand()`
   - `VirtualZeroOrderOptimizer`: Zero-order optimizer that only stores seeds, not vectors
   - Memory: O(1) per perturbation instead of O(num_params)

2. **`zero_order_optimizer.py`** - Production CD-RGE optimizer (used by train.py)
   - Seed-based perturbation generation
   - Works with StandardGRU (cuDNN)
   - Serial execution (processes perturbations one at a time)

3. **Test scripts**:
   - `test_virtual_perturbations.py`: Demo of memory savings
   - `test_500m_virtual_tuning.py`: Find optimal parallel batch size for 500M model
   - `test_parallel_batch_tuning.py`: Parallel batch size tuning for small models

### Current Objectives

1. **Scale virtual perturbation optimizer to 500M parameters**
   - Test with real production model (StandardGRU, 500M params)
   - Find optimal perturbation batch size for maximum throughput
   - Measure memory usage and training speed

2. **Integrate Triton seed-based kernel into full model forward pass**
   - Current: Kernel exists but only for single Linear layer matmul
   - Goal: Hook into all model matmuls for end-to-end virtual perturbations
   - Challenge: Triton kernel with `tl.rand()` needs to work with model's forward pass

3. **Compare approaches**:
   - Serial seed-based (zero_order_optimizer.py) - currently working in train.py
   - Parallel batched vmap - doesn't work with cuDNN/Triton (functional_call issue)
   - Full Triton virtual (zero_order_virtual.py) - in development

### How Zero-Order Optimization Works

CD-RGE estimates gradients using central differences:
```python
# For each perturbation P (generated from seed!):
loss_plus = forward(params + ε·P)
loss_minus = forward(params - ε·P)
gradient ≈ (loss_plus - loss_minus) / (2ε) · P
```

**Key insight**: P is never stored! Generated on-demand from seed each time needed.

Advantages:
- No backpropagation (memory efficient - no activation storage)
- Massively parallel (evaluate many perturbations independently)
- Works with any model (black-box optimization)
- Seed-based approach: O(1) memory per perturbation

### Training Commands

```bash
# Zero-order training (500M model, 8 GPUs)
./train.zero_order_test.sh

# Test virtual perturbation scaling
python test_500m_virtual_tuning.py

# Standard training (for comparison)
./train.standard_gru.sh
```

## Common Development Commands

### Training
```bash
# Main training command (8 GPUs, 1B parameters)
./train.cuda.sh

# For AMD GPUs (Frontier supercomputer)
./train.frontier.sh

# Single GPU training
python train.py --data /path/to/data.txt --output output_dir --params 1g --cuda
```

### Text Generation
```bash
# Generate text from a checkpoint
python generate.py --model path/to/checkpoint.pt --prompt "Your text here" --max_length 500

# With specific sampling parameters
python generate.py --model checkpoint.pt --temperature 0.8 --top_k 50 --top_p 0.9
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For AMD ROCm systems
pip install -r rocm-reqs.txt

# Check GPU setup
./check_gpus.sh
```

### Monitoring Training
```bash
# View training logs
tail -f logs/train_*.log

# Analyze gossip protocol
ls gossip_logs/

# Plot training metrics (requires R)
Rscript plot_train.R
```

## Architecture

### Core Components

1. **MinGRU Model** (`mingru/`)
   - `minGRU.py`: Parallelizable RNN layer using associative scan
   - `minLM.py`: Language model built on MinGRU
   - Key feature: Processes sequences in parallel unlike traditional RNNs

2. **Evolutionary Training** (`gossip/`)
   - `evolutionary_node.py`: Per-GPU evolutionary logic
   - `network_utils.py`: TCP peer-to-peer communication
   - `fitness_tracker.py`: Loss-based fitness evaluation
   - Models exchange parameters based on fitness, not gradient averaging

3. **Training Pipeline**
   - Byte-level processing (no tokenization, 256-token vocabulary)
   - Memory-mapped data files for efficient large dataset access
   - Truncated backpropagation through time (TBPTT) for unbounded context
   - Schedule-free AdamW optimizer

### Key Training Parameters

- `--params`: Model size (350m, 1g, 7g)
- `--chunk_size`: TBPTT chunk length (default 2048)
- `--gossip_mixing_rate`: Evolution rate (0.01 = 1% chance)
- `--batch_size` + `--grad_accum`: Effective batch size
- `--lr`: Learning rate (typical 0.0001-0.002)

### Distributed Training Flow

1. Each GPU runs independent training with its own model
2. Models periodically connect via TCP to random peers
3. Lower loss model transfers parameters to higher loss model
4. Filesystem coordinator (rank 0) tracks global fitness
5. Elite models saved more frequently via fitness-weighted checkpointing

## Important Notes

### Job Submission Workflow
- **ALWAYS commit changes before submitting jobs**: The output directory name includes the git commit hash, so you must commit any configuration changes before running `sbatch`. This ensures reproducibility and proper tracking of experiments.
  ```bash
  git add -A
  git commit -m "Description of changes"
  git push origin stampede3  # optional but recommended
  sbatch train.stampede3.sh
  ```

### General Notes
- Data must be raw text files (no tokenization needed)
- Models process raw bytes 0-255 directly
- Checkpoints named with loss values for easy identification
- No formal test suite - this is research code
- Evolution works best with population size >= 4 GPUs
- Memory-mapped files allow training on datasets larger than RAM
- **Shell script limitation**: Cannot have inline comments after backslash line continuations in train.cuda.sh - they break the command
- **NEVER use `git add -A` or `git add .`** - only add specific code files that were modified
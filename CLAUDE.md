# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

gruboros is a research platform for training large language models using evolutionary optimization with MinGRU (efficient RNN) architecture and byte-level modeling. Key innovation: replaces traditional synchronized distributed training with evolutionary fitness-based parameter exchange between models.

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

- Data must be raw text files (no tokenization needed)
- Models process raw bytes 0-255 directly
- Checkpoints named with loss values for easy identification
- No formal test suite - this is research code
- Evolution works best with population size >= 4 GPUs
- Memory-mapped files allow training on datasets larger than RAM
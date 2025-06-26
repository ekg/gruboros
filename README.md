# gruboros: Evolutionary MinGRU Language Model Training

**Author**: Erik Garrison  
**License**: MIT  
**Funding**: Oak Ridge Leadership Computing Facility Director's Discretionary Award BIF148

This research platform explores a fundamentally different approach to training large language models by combining evolutionary optimization with efficient RNN architectures and universal byte-level modeling. Rather than relying on transformer architectures that require massive computational resources and complex tokenization schemes, gruboros demonstrates that recurrent neural networks—when properly parallelized—can achieve competitive performance while offering unique advantages in memory efficiency, context handling, and data processing flexibility.

The core insight driving this work is that current large language model training faces several fundamental bottlenecks: transformers scale poorly with sequence length, traditional distributed training suffers from communication overhead that grows quadratically with model size, and tokenization creates artificial boundaries that may limit model understanding. By addressing these challenges simultaneously through evolutionary training protocols, efficient RNN architectures, and raw byte processing, gruboros opens new possibilities for both model efficiency and training at scale.

## Technical Innovations

**MinGRU Architecture**: The foundation is built on MinGRU, a linearized gated recurrent unit from the "Were RNNs All We Needed?" paper that enables parallel training of RNNs through associative scan operations. Unlike traditional RNNs that must be trained sequentially, MinGRU can leverage modern GPU parallelism while maintaining the memory efficiency and unbounded context capabilities that make RNNs attractive for sequence modeling. The log-space implementation ensures numerical stability with positive hidden states, enabling training of billion-parameter models on individual GPUs.

**Evolutionary Training Protocol**: Instead of traditional distributed data parallel training with its communication bottlenecks, gruboros implements a "gossip protocol" where models evolve through fitness-based parameter exchange. This approach circumvents the diminishing returns of lock-step gradient accumulation and averaging—the evolution rate scales proportionally with population size rather than suffering from quadratic communication overhead. Models exchange parameters based on training loss fitness scores, with multiple merge strategies including clonal replacement, weighted recombination, and optimizer state interpolation. Elite preservation maintains populations of best-performing models while rejuvenation mechanisms prevent population stagnation.

**Universal Byte-Level Modeling**: Rather than preprocessing data through tokenizers that impose linguistic assumptions, gruboros processes raw bytes directly using a 256-token vocabulary corresponding to all possible byte values. This universal approach enables learning from any data format—text, code, binary files, or arbitrary data—without preprocessing bottlenecks or tokenization artifacts. Memory-mapped files provide efficient access to large datasets without loading everything into RAM.

**Chunked Training with Unbounded Context**: The system implements truncated backpropagation through time (TBPTT) to process arbitrarily long sequences in manageable chunks while maintaining hidden state continuity across chunks. This enables effective long-context modeling without the quadratic memory growth that plagues transformer architectures. Configurable chunk sizes and context windows allow adaptation to different sequence lengths and memory constraints.

**Distributed Optimization**: The platform integrates schedule-free optimization (Schedule-Free AdamW) for stable training without learning rate schedules, gradient accumulation for large effective batch sizes, automatic mixed precision training, and fitness-weighted checkpointing that saves high-performing models more frequently.

## OLCF Distributed Training

The evolutionary gossip protocol implements a sophisticated peer-to-peer network architecture designed for large-scale HPC environments like Oak Ridge Leadership Computing Facility. Each training process runs both a TCP server and client, enabling true all-to-all communication where any process can connect to any other process in the distributed job. The system automatically discovers all peer processes using SLURM job node lists and creates a complete connectivity graph across all nodes and GPUs.

The protocol works through fitness-based tournaments: when two processes connect, they exchange their current training loss values, and the process with better fitness (lower loss) sends its weights to the other process. This creates evolutionary pressure where successful training configurations spread through the population. Unlike traditional distributed data parallel training that requires synchronous all-reduce operations with quadratic communication overhead, this approach scales linearly with population size since each process only needs to communicate with randomly selected peers.

The filesystem component provides optional coordination support - Rank 0 maintains global fitness rankings for bookkeeping while all processes participate in decentralized evolution decisions through direct peer-to-peer TCP connections. This hybrid approach combines the efficiency of peer-to-peer communication with optional centralized monitoring, providing fault tolerance that survives node failures and scheduler interruptions.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Memory Mapping  │───▶│  Byte Chunks    │
│  (Any Format)   │    │   (No Tokenizer) │    │   (0-255)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Evolutionary    │◀───│    MinGRU LM     │◀───│ Chunked TBPTT   │
│ Gossip Protocol │    │   Architecture    │    │   Training      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Project Structure

```
gruboros/
├── train.py                 # Main training script with evolutionary protocol
├── train.cuda.sh           # Training launcher script (state-of-the-art setup)
├── mingru/                 # MinGRU model implementation
│   ├── minGRU.py           # Core MinGRU layer with associative scan
│   └── minLM.py            # MinGRU-based language model
├── gossip/                 # Evolutionary training system
│   ├── evolutionary_node.py    # Per-node evolutionary logic
│   ├── filesystem_coordinator.py # Population-wide coordination
│   ├── fitness_tracker.py      # Training loss tracking
│   └── network_utils.py        # Distributed communication utilities
├── generate.py             # Text generation script
└── requirements.txt        # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- DeepSpeed (used as launcher only)

### Installation
```bash
git clone <repository-url>
cd gruboros
pip install -r requirements.txt
```

### Training
The current state-of-the-art training configuration is in `train.cuda.sh`:

```bash
# Edit data path in train.cuda.sh to point to your dataset
# Current setup uses /mnt/nvme1n1/erikg/fineweb-edu/sample/350BT.txt
./train.cuda.sh
```

### Key Training Parameters

- **Model Size**: `--params 1g` (1 billion parameters)
- **Context**: `--chunk_size 2048 --context_chunks 16` (32K effective context)
- **Batch Size**: `--batch_size 1 --grad_accum 16` (effective batch size 16)
- **Evolution**: `--gossip_mixing_rate 0.01` (1% mixing probability)
- **Learning Rate**: `--lr 0.002` with Schedule-Free optimizer

## Training Features

### Multi-GPU Distributed Training
The current implementation supports 8-GPU setups using the GLOO backend, where each GPU trains independently with parameter sharing through the gossip communication protocol. Models exchange parameters based on fitness rather than traditional gradient averaging.

### Checkpoint Management
The system implements fitness-based checkpoint naming that includes loss values in filenames, maintains multiple best-performing checkpoints for elite preservation, and uses a background thread for automatic cleanup and disk space management. Symlinks to the latest and best models provide easy access for inference and analysis.

### Evolutionary Protocol
Each GPU maintains a model in the population with fitness evaluation based on exponential moving average of training loss. Selection pressure ensures better models have higher probability of being chosen for parameter exchange, while rejuvenation mechanisms maintain diversity and prevent premature convergence.

## Advanced Features

### Filesystem Coordination
The system implements a hybrid coordination approach where Rank 0 maintains global fitness rankings for monitoring and bookkeeping while all processes participate in decentralized evolution through direct TCP peer-to-peer connections. Each process runs both a server (listening on unique ports) and client (connecting to randomly selected peers) to enable true all-to-all communication. The filesystem component provides optional support for processes that fall behind by maintaining serialized models on disk, but the core evolutionary mechanism relies on direct network communication between processes.

### Memory Optimization
The system uses memory-mapped data access so large datasets don't require full RAM loading, implements chunked processing that maintains constant memory usage regardless of sequence length, and employs gradient checkpointing to reduce memory usage during backpropagation.

### Monitoring and Logging
Training generates structured JSON-formatted logs for analysis, tracking loss, learning rate, and gossip events. Tools are provided for population fitness visualization and analysis.

## Research Applications

This codebase enables research in:
- **Evolutionary Neural Architecture Search**
- **Long-Context Language Modeling**
- **Byte-Level Language Understanding**
- **Distributed Optimization**
- **RNN Renaissance and Efficiency**

## Performance Characteristics

- **Single GPU Training**: 1B+ parameter models on individual GPUs
- **Parallel RNN Training**: Parallel training of recurrent architectures
- **Long Context**: Handles sequences much longer than traditional transformers
- **Raw Data Learning**: No preprocessing bottlenecks or tokenization artifacts

## Citation

If you use this code in your research, please cite the MinGRU paper:
```
@article{feng2024were,
  title={Were RNNs All We Needed?},
  author={Feng, Leo and Tung, Frederick and Ahmed, Mohamed Osama and Bengio, Yoshua and Hajimirsadeghi, Hossein},
  journal={arXiv preprint arXiv:2410.01201},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions welcome. Please open issues or pull requests for bugs, features, or improvements.

---

*This project represents a convergence of several research directions: efficient RNN architectures, evolutionary optimization, byte-level modeling, and distributed training. The result is a unique platform for exploring the intersection of these techniques in large-scale language model development.*
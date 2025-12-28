"""
Elman Ablation Ladder - Log-Space Experiment Series

This module implements the systematic ablation from stock Elman to full
log-space Triple R, following docs/LOG_SPACE_IMPLEMENTATION_HANDOFF.md

Ablation Ladder:
    Level 0: Stock Elman - Basic tanh recurrence
    Level 1: Gated Elman - + Input-dependent delta gate
    Level 2: Selective Elman - + compete×silu output
    Level 3: Diagonal Selective - Diagonal r_h (like Mamba2's diagonal A)
    Level 4: Log-Storage Diagonal - + signed log storage for hidden state
    Level 5: Log-Compute Full - Full R via logsumexp decomposition
    Level 6: Triple R - + R_delta modulation

Goal: Find the simplest level that matches Mamba2 (3.924 avg50).

Usage:
    from mingru.elman_ladder import StockElman, GatedElman, SelectiveElman
    from mingru.elman_ladder import DiagonalSelective, LogStorageDiagonal
    from mingru.elman_ladder import LogComputeFull, LogSpaceTripleR

    # Each module exports:
    # - Core cell class (e.g., StockElmanCell)
    # - Full layer class (e.g., StockElmanLayer)
    # - HASTE_AVAILABLE flag indicating if CUDA kernel is available
"""

# Import ladder levels
try:
    from .stock_elman import StockElman, StockElmanCell, LEVEL_0_AVAILABLE
except ImportError:
    StockElman = None
    StockElmanCell = None
    LEVEL_0_AVAILABLE = False

try:
    from .gated_elman import GatedElman, GatedElmanCell, LEVEL_1_AVAILABLE
except ImportError:
    GatedElman = None
    GatedElmanCell = None
    LEVEL_1_AVAILABLE = False

try:
    from .selective_elman import SelectiveElman, SelectiveElmanCell, LEVEL_2_AVAILABLE
except ImportError:
    SelectiveElman = None
    SelectiveElmanCell = None
    LEVEL_2_AVAILABLE = False

try:
    from .diagonal_selective import DiagonalSelective, DiagonalSelectiveCell, LEVEL_3_AVAILABLE
except ImportError:
    DiagonalSelective = None
    DiagonalSelectiveCell = None
    LEVEL_3_AVAILABLE = False

try:
    from .log_storage_diagonal import LogStorageDiagonal, LogStorageDiagonalCell, LEVEL_4_AVAILABLE
except ImportError:
    LogStorageDiagonal = None
    LogStorageDiagonalCell = None
    LEVEL_4_AVAILABLE = False

try:
    # Level 5: Log-space storage with hybrid compute (cuBLAS matmuls)
    from .log_compute_full import LogComputeFull, LogComputeFullCell, LEVEL_5_AVAILABLE
except ImportError:
    LogComputeFull = None
    LogComputeFullCell = None
    LEVEL_5_AVAILABLE = False

try:
    # Level 5 FAST: Pure linear (no log-space at all)
    from .log_compute_full_fast import FullRElman, FullRElmanCell, LEVEL_5_FAST_AVAILABLE
except ImportError:
    FullRElman = None
    FullRElmanCell = None
    LEVEL_5_FAST_AVAILABLE = False

try:
    # Level 6: Log-space storage with hybrid compute (cuBLAS matmuls)
    from .logspace_triple_r import LogSpaceTripleR, LogSpaceTripleRCell, LEVEL_6_AVAILABLE
except ImportError:
    LogSpaceTripleR = None
    LogSpaceTripleRCell = None
    LEVEL_6_AVAILABLE = False

try:
    # Level 6 FAST: Pure linear (no log-space at all)
    from .logspace_triple_r_fast import TripleRElman, TripleRElmanCell, LEVEL_6_FAST_AVAILABLE
except ImportError:
    TripleRElman = None
    TripleRElmanCell = None
    LEVEL_6_FAST_AVAILABLE = False

# Language model wrapper
try:
    from .ladder_lm import LadderLM, create_ladder_model
except ImportError:
    LadderLM = None
    create_ladder_model = None


def get_available_levels():
    """Return dict of available ladder levels."""
    return {
        0: ("Stock Elman", LEVEL_0_AVAILABLE, StockElman),
        1: ("Gated Elman", LEVEL_1_AVAILABLE, GatedElman),
        2: ("Selective Elman", LEVEL_2_AVAILABLE, SelectiveElman),
        3: ("Diagonal Selective", LEVEL_3_AVAILABLE, DiagonalSelective),
        4: ("Log-Storage Diagonal", LEVEL_4_AVAILABLE, LogStorageDiagonal),
        5: ("Log-Compute Full (Hybrid)", LEVEL_5_AVAILABLE, LogComputeFull),
        6: ("Log-Space Triple R (Hybrid)", LEVEL_6_AVAILABLE, LogSpaceTripleR),
        # Fast variants (no log-space storage)
        "5-fast": ("Full R Linear", LEVEL_5_FAST_AVAILABLE, FullRElman),
        "6-fast": ("Triple R Linear", LEVEL_6_FAST_AVAILABLE, TripleRElman),
    }


def get_ladder_level(level):
    """Get the module class for a specific ladder level."""
    levels = get_available_levels()
    if level not in levels:
        raise ValueError(f"Invalid level {level}. Must be 0-6.")
    name, available, cls = levels[level]
    if not available:
        raise ImportError(f"Level {level} ({name}) is not available. Check Haste installation.")
    return cls


__all__ = [
    # Level classes (with log-space storage)
    'StockElman', 'StockElmanCell',
    'GatedElman', 'GatedElmanCell',
    'SelectiveElman', 'SelectiveElmanCell',
    'DiagonalSelective', 'DiagonalSelectiveCell',
    'LogStorageDiagonal', 'LogStorageDiagonalCell',
    'LogComputeFull', 'LogComputeFullCell',
    'LogSpaceTripleR', 'LogSpaceTripleRCell',
    # Fast variants (no log-space)
    'FullRElman', 'FullRElmanCell',
    'TripleRElman', 'TripleRElmanCell',
    # Availability flags
    'LEVEL_0_AVAILABLE', 'LEVEL_1_AVAILABLE', 'LEVEL_2_AVAILABLE',
    'LEVEL_3_AVAILABLE', 'LEVEL_4_AVAILABLE', 'LEVEL_5_AVAILABLE',
    'LEVEL_6_AVAILABLE', 'LEVEL_5_FAST_AVAILABLE', 'LEVEL_6_FAST_AVAILABLE',
    # Language model
    'LadderLM', 'create_ladder_model',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]

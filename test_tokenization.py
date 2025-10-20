#!/usr/bin/env python3
"""Quick test to verify tokenization in dataset."""

import sys
import torch
sys.path.insert(0, '.')

from mingru.tokenizers import get_tokenizer
from train import DocumentStreamDataset

# Test tokenizer
print("=== Testing Tokenizer ===")
tokenizer = get_tokenizer('tiktoken', encoding_name='cl100k_base')
print(f"Tokenizer: {tokenizer}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Test encoding
test_text = "Hello, world! This is a test of the tokenization system."
tokens = tokenizer.encode(test_text)
print(f"\nTest text: {test_text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Token range: [{min(tokens)}, {max(tokens)}]")
print(f"Decoded: {tokenizer.decode(tokens)}")

# Test dataset
print("\n=== Testing Dataset ===")
dataset = DocumentStreamDataset(
    "/mnt/nvme2n1/erikg/pile.txt",  # First positional arg
    chunk_size=256,  # Small for testing
    seed=42,
    rank=0,
    world_size=1,
    tokenizer=tokenizer
)

print(f"\nDataset tokenizer: {dataset.tokenizer}")
print(f"Dataset tokenizer class: {dataset.tokenizer.__class__.__name__}")

# Get a few chunks
print("\n=== Getting chunks ===")
for i in range(3):
    chunk, is_doc_end, actual_len = dataset.get_next_chunk()
    print(f"\nChunk {i}:")
    print(f"  Shape: {chunk.shape}")
    print(f"  Actual length: {actual_len}")
    print(f"  Is doc end: {is_doc_end}")
    print(f"  Token range: [{chunk.min().item()}, {chunk.max().item()}]")
    print(f"  First 10 tokens: {chunk[:10].tolist()}")

    # Try to decode a sample
    non_zero = chunk[chunk > 0][:50]  # First 50 non-padding tokens
    if len(non_zero) > 0:
        decoded = tokenizer.decode(non_zero.tolist())
        print(f"  Decoded sample: {decoded[:100]}...")

print("\n=== Test Complete ===")

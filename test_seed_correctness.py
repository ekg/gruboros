#!/usr/bin/env python3
"""
Test numerical correctness of seed-based perturbation generation

CRITICAL: The PRNG must unroll identically whether we:
1. Materialize the full vector: torch.randn(N)
2. Generate in chunks
3. Generate element-by-element

This test verifies all approaches produce IDENTICAL random sequences.
"""

import torch
import numpy as np

def test_basic_seed_correctness():
    """Test that torch.manual_seed gives consistent results"""
    print("="*80)
    print("TEST 1: Basic seed correctness")
    print("="*80)

    seed = 42
    num_params = 1000

    # Approach 1: Materialize full vector
    torch.manual_seed(seed)
    materialized = torch.randn(num_params)

    # Approach 2: Materialize again with same seed
    torch.manual_seed(seed)
    materialized_again = torch.randn(num_params)

    max_diff = torch.max(torch.abs(materialized - materialized_again)).item()

    print(f"\nMaterialized twice with seed {seed}:")
    print(f"  Max difference: {max_diff}")
    print(f"  Identical: {torch.allclose(materialized, materialized_again)}")

    if max_diff == 0:
        print("✓ PASS: Seeds produce identical results")
    else:
        print("✗ FAIL: Seeds produce different results!")
        return False

    return True


def test_chunked_generation():
    """Test that generating in chunks matches full generation"""
    print("\n" + "="*80)
    print("TEST 2: Chunked generation correctness")
    print("="*80)

    seed = 42
    num_params = 1000
    chunk_size = 100

    # Approach 1: Materialize full vector
    torch.manual_seed(seed)
    materialized = torch.randn(num_params)

    # Approach 2: Generate in chunks
    torch.manual_seed(seed)
    chunked = []
    for i in range(0, num_params, chunk_size):
        chunk = torch.randn(min(chunk_size, num_params - i))
        chunked.append(chunk)
    chunked = torch.cat(chunked)

    max_diff = torch.max(torch.abs(materialized - chunked)).item()

    print(f"\nGenerated in {num_params // chunk_size} chunks of {chunk_size}:")
    print(f"  Max difference: {max_diff}")
    print(f"  Identical: {torch.allclose(materialized, chunked)}")

    if max_diff == 0:
        print("✓ PASS: Chunked generation matches materialized")
    else:
        print("✗ FAIL: Chunked generation differs!")
        print(f"  First 10 materialized: {materialized[:10]}")
        print(f"  First 10 chunked:      {chunked[:10]}")
        return False

    return True


def test_torch_generator_approach():
    """Test using torch.Generator for explicit state management"""
    print("\n" + "="*80)
    print("TEST 3: torch.Generator approach")
    print("="*80)

    seed = 42
    num_params = 1000

    # Approach 1: Materialize with manual_seed
    torch.manual_seed(seed)
    materialized = torch.randn(num_params)

    # Approach 2: Use Generator explicitly
    generator = torch.Generator()
    generator.manual_seed(seed)
    generated = torch.randn(num_params, generator=generator)

    max_diff = torch.max(torch.abs(materialized - generated)).item()

    print(f"\nUsing torch.Generator with seed {seed}:")
    print(f"  Max difference: {max_diff}")
    print(f"  Identical: {torch.allclose(materialized, generated)}")

    if max_diff == 0:
        print("✓ PASS: Generator produces identical results")
    else:
        print("✗ FAIL: Generator produces different results!")
        return False

    return True


def test_per_parameter_seed_approach():
    """
    Test if we can use per-parameter seeding (WRONG APPROACH - will fail!)

    This is what zero_order_virtual.py currently does in the Triton kernel.
    It will NOT match materialized generation!
    """
    print("\n" + "="*80)
    print("TEST 4: Per-parameter seeding (EXPECTED TO FAIL)")
    print("="*80)

    seed = 42
    num_params = 1000

    # Approach 1: Materialize with single seed
    torch.manual_seed(seed)
    materialized = torch.randn(num_params)

    # Approach 2: Per-parameter seeds (WRONG!)
    per_param_seeded = torch.zeros(num_params)
    for i in range(num_params):
        torch.manual_seed(seed + i)  # Different seed per parameter!
        per_param_seeded[i] = torch.randn(1).item()

    max_diff = torch.max(torch.abs(materialized - per_param_seeded)).item()
    mean_diff = torch.mean(torch.abs(materialized - per_param_seeded)).item()

    print(f"\nPer-parameter seeds (seed + index):")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Identical: {torch.allclose(materialized, per_param_seeded)}")

    if max_diff > 0.01:  # Expect large difference
        print("✓ EXPECTED: Per-parameter seeding does NOT match materialized")
        print("  → This approach is numerically incorrect!")
        print(f"  First 10 materialized:    {materialized[:10]}")
        print(f"  First 10 per-param-seed:  {per_param_seeded[:10]}")
    else:
        print("✗ UNEXPECTED: Per-parameter seeding somehow matched?")
        return False

    return True


def test_correct_seed_based_approach():
    """
    CORRECT approach: Use generator and generate full sequence

    This is what we should implement:
    - Create generator with seed
    - Generate full parameter vector from generator
    - Don't store it, just use it immediately
    """
    print("\n" + "="*80)
    print("TEST 5: Correct seed-based approach for virtual perturbations")
    print("="*80)

    seed = 42
    num_params = 1000

    # Approach 1: Materialize
    torch.manual_seed(seed)
    materialized = torch.randn(num_params)

    # Approach 2: Generate but don't store (simulate virtual generation)
    def generate_virtual_perturbation(seed, num_params, apply_fn):
        """Generate perturbation and immediately apply to parameters"""
        generator = torch.Generator()
        generator.manual_seed(seed)
        perturbation = torch.randn(num_params, generator=generator)
        # In real implementation, we'd apply here and then discard
        return perturbation  # Just for verification

    virtual = generate_virtual_perturbation(seed, num_params, lambda p: p)

    max_diff = torch.max(torch.abs(materialized - virtual)).item()

    print(f"\nVirtual generation (generate + immediate use):")
    print(f"  Max difference: {max_diff}")
    print(f"  Identical: {torch.allclose(materialized, virtual)}")

    if max_diff == 0:
        print("✓ PASS: Virtual generation is numerically identical!")
        print("\n  This is the CORRECT approach:")
        print("  1. Create generator from seed")
        print("  2. Generate full perturbation vector")
        print("  3. Apply to parameters IMMEDIATELY")
        print("  4. Discard perturbation (never store)")
        print("  → Memory: O(1) per perturbation (just seed)")
        print("  → Numerics: Exact match to materialized")
    else:
        print("✗ FAIL: Virtual generation differs!")
        return False

    return True


def test_rademacher_perturbations():
    """Test Rademacher (±1) perturbations maintain correctness"""
    print("\n" + "="*80)
    print("TEST 6: Rademacher perturbations")
    print("="*80)

    seed = 42
    num_params = 1000

    # Approach 1: Materialize and convert to Rademacher
    torch.manual_seed(seed)
    materialized_gaussian = torch.randn(num_params)
    materialized_rademacher = torch.sign(materialized_gaussian)

    # Approach 2: Virtual generation with Rademacher
    generator = torch.Generator()
    generator.manual_seed(seed)
    virtual_gaussian = torch.randn(num_params, generator=generator)
    virtual_rademacher = torch.sign(virtual_gaussian)

    max_diff = torch.max(torch.abs(materialized_rademacher - virtual_rademacher)).item()

    print(f"\nRademacher perturbations (sign(randn)):")
    print(f"  Max difference: {max_diff}")
    print(f"  Identical: {torch.allclose(materialized_rademacher, virtual_rademacher)}")

    # Check statistics
    mat_pos_frac = (materialized_rademacher > 0).float().mean()
    virt_pos_frac = (virtual_rademacher > 0).float().mean()

    print(f"  Materialized: {mat_pos_frac:.3f} positive")
    print(f"  Virtual:      {virt_pos_frac:.3f} positive")

    if max_diff == 0:
        print("✓ PASS: Rademacher perturbations match exactly")
    else:
        print("✗ FAIL: Rademacher perturbations differ!")
        return False

    return True


if __name__ == '__main__':
    print("="*80)
    print("SEED-BASED PERTURBATION: NUMERICAL CORRECTNESS TESTS")
    print("="*80)
    print("\nThese tests verify that seed-based perturbation generation")
    print("produces IDENTICAL results to materialized generation.")
    print("\nCritical requirement: PRNG must unroll identically!")

    all_passed = True

    all_passed &= test_basic_seed_correctness()
    all_passed &= test_chunked_generation()
    all_passed &= test_torch_generator_approach()
    all_passed &= test_per_parameter_seed_approach()
    all_passed &= test_correct_seed_based_approach()
    all_passed &= test_rademacher_perturbations()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nCorrect approach for virtual perturbations:")
        print("  1. Store only seed (4 bytes)")
        print("  2. Create torch.Generator from seed")
        print("  3. Generate full perturbation vector")
        print("  4. Apply to parameters immediately")
        print("  5. Discard perturbation vector")
        print("\n  → Memory: O(1) per perturbation")
        print("  → Numerics: Exact match to materialized")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nReview failed tests above.")

    print("="*80)

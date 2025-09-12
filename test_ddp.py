#!/usr/bin/env python3
"""
Test script to verify DDP implementation in train.py
"""

import subprocess
import sys
import os

def test_import():
    """Test that the modified train.py can be imported without errors"""
    print("Testing import of train.py...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train
        print("✓ train.py imports successfully")
        
        # Check that new arguments exist
        from train import get_args
        args = get_args()
        parser = args.__class__.__module__
        print("✓ get_args() function works")
        
        # Check for DDP-related functions
        if hasattr(train, 'setup_ddp_groups'):
            print("✓ setup_ddp_groups function found")
        else:
            print("✗ setup_ddp_groups function not found")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_help():
    """Test that train.py --help works and shows DDP options"""
    print("\nTesting train.py --help...")
    try:
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"✗ Help command failed with return code {result.returncode}")
            return False
            
        # Check for DDP-related options in help text
        help_text = result.stdout
        ddp_options = ["--ddp", "--ddp-find-unused"]
        
        for option in ddp_options:
            if option in help_text:
                print(f"✓ Found {option} in help text")
            else:
                print(f"✗ {option} not found in help text")
                return False
                
        return True
    except Exception as e:
        print(f"✗ Help test failed: {e}")
        return False

def test_syntax():
    """Test that the Python syntax is valid"""
    print("\nTesting Python syntax...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "train.py"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ Python syntax is valid")
            return True
        else:
            print(f"✗ Syntax errors found:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Syntax test failed: {e}")
        return False

def main():
    print("DDP Implementation Test Suite")
    print("=" * 40)
    
    tests = [
        ("Syntax Check", test_syntax),
        ("Import Test", test_import),
        ("Help Test", test_help),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("-" * 40)
    
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("✓ All tests passed! DDP implementation looks good.")
        print("\nNext steps:")
        print("1. Test with a small dataset using single node DDP:")
        print("   torchrun --nproc_per_node=2 train.py --ddp --data small_data.txt --params 100m ...")
        print("\n2. Test multi-node DDP with SLURM:")
        print("   srun --nodes=2 --ntasks-per-node=4 python train.py --ddp --data data.txt ...")
    else:
        print("✗ Some tests failed. Please review the implementation.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
This script patches DeepSpeed to better support ROCm by monkey-patching
the CUDA version detection to return a compatible version for AMD GPUs.
"""

import os
import sys
from pathlib import Path

# Find DeepSpeed installed location
import deepspeed
deepspeed_path = Path(deepspeed.__file__).parent

def patch_deepspeed_files():
    # Path to the builder.py file that contains the CUDA version detection
    builder_path = deepspeed_path / "ops" / "op_builder" / "builder.py"
    
    if not builder_path.exists():
        print(f"Error: Could not find {builder_path}")
        return False
    
    # Read the file
    with open(builder_path, 'r') as f:
        content = f.read()
    
    # Look for the installed_cuda_version function
    if "def installed_cuda_version():" not in content:
        print("Error: Could not find installed_cuda_version function in builder.py")
        return False
    
    # Create a backup
    backup_path = builder_path.with_suffix('.py.bak')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
    
    # Replace the function with our patched version
    patched_content = content.replace(
        "def installed_cuda_version():",
        """def installed_cuda_version():
    # ROCm compatibility - return a compatible CUDA version
    if os.environ.get('ROCM_HOME') or os.environ.get('DS_SKIP_CUDA_CHECK'):
        print("DeepSpeed: Using ROCm compatibility mode - reporting CUDA 11.0")
        return 11, 0
"""
    )
    
    # Only write if we actually made changes
    if patched_content != content:
        with open(builder_path, 'w') as f:
            f.write(patched_content)
        print(f"Patched {builder_path} for ROCm compatibility")
        return True
    else:
        print("No changes needed - already patched or incompatible format")
        return False

def main():
    print(f"DeepSpeed patch tool for ROCm compatibility")
    print(f"DeepSpeed path: {deepspeed_path}")
    
    # ROCm environment check
    if os.environ.get('ROCM_HOME'):
        print(f"Found ROCM_HOME: {os.environ.get('ROCM_HOME')}")
    else:
        print("Warning: ROCM_HOME not set in environment")
    
    # Patch the files
    success = patch_deepspeed_files()
    
    if success:
        print("\nPatch applied successfully! DeepSpeed should now work with ROCm.")
        print("Run your training script as normal.")
    else:
        print("\nPatch could not be applied. You may need to manually modify DeepSpeed.")

if __name__ == "__main__":
    main()

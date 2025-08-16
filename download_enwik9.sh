#!/bin/bash
# download_enwik9.sh - Download and prepare enwik9 dataset

set -e

echo "=== Downloading enwik9 dataset ==="

# Create directory
mkdir -p "$SCRATCH/train"
cd "$SCRATCH/train"

# Check if already exists
if [ -f "enwik9" ]; then
    echo "enwik9 already exists at $SCRATCH/train/enwik9"
    ls -lh enwik9
    exit 0
fi

# Download
echo "Downloading enwik9.zip..."
wget http://mattmahoney.net/dc/enwik9.zip

# Extract
echo "Extracting..."
unzip enwik9.zip

# Cleanup
rm enwik9.zip

# Show info
echo ""
echo "=== Download complete ==="
ls -lh enwik9
echo "Dataset location: $SCRATCH/train/enwik9"
echo "Size: $(du -h enwik9 | cut -f1)"
echo ""
echo "enwik9 is the first 10^9 bytes of English Wikipedia (UTF-8)"
#!/bin/bash
# Quick script to submit the gruboros training job
# Run this from a login node

echo "Submitting gruboros 100M model training job..."
cd /work2/09647/erikgarrison/stampede3/gruboros
sbatch train.stampede3.sh
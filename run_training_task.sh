#!/bin/bash
# Wrapper script for each srun task
# SLURM will set SLURM_PROCID and SLURM_LOCALID for each task

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Task starting: RANK=$RANK LOCAL_RANK=$LOCAL_RANK on $(hostname)"

python train.py "$@"
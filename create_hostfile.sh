#!/bin/bash
# Create hostfile for DeepSpeed from Slurm allocation

# Get list of nodes
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt

echo "Creating hostfile for DeepSpeed..."
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

echo "Created hostfile with the following contents:"
cat $HOSTFILE

echo "Hostfile is ready at: $HOSTFILE"

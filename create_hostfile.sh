#!/bin/bash
# Create hostfile for DeepSpeed from Slurm allocation

# Get list of unique nodes
HOSTFILE=hostfile.txt

echo "Creating hostfile for DeepSpeed..."
# Get unique hostnames using scontrol
scontrol show hostnames $SLURM_JOB_NODELIST > unique_hosts_$SLURM_JOB_ID
# Add slot information - each host gets 8 slots (for 8 GPUs)
sed 's/$/ slots=8/' unique_hosts_$SLURM_JOB_ID > $HOSTFILE
# Clean up temporary file
rm -f unique_hosts_$SLURM_JOB_ID

echo "Created hostfile with the following contents:"
cat $HOSTFILE

echo "Hostfile is ready at: $HOSTFILE"

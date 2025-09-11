#!/bin/bash
# Real-time monitoring of z-gate statistics during training
# Usage: ./monitor_z_gates.sh [output_dir]
# If no output_dir provided, automatically finds the latest in /mnt/nvme2n1/erikg/minlms/

# Base directory where training outputs are stored
BASE_DIR="/mnt/nvme2n1/erikg/minlms"

# If argument provided, use it; otherwise find latest directory
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    # Find the most recently created directory in BASE_DIR
    if [ -d "$BASE_DIR" ]; then
        OUTPUT_DIR=$(ls -dt "$BASE_DIR"/*/ 2>/dev/null | head -n 1)
        if [ -z "$OUTPUT_DIR" ]; then
            echo "No output directories found in $BASE_DIR"
            exit 1
        fi
        echo "Auto-detected latest output directory: $OUTPUT_DIR"
    else
        echo "Base directory $BASE_DIR not found"
        exit 1
    fi
fi

TSV_FILE="${OUTPUT_DIR}/z_stats_rank0.tsv"

if [ ! -f "$TSV_FILE" ]; then
    echo "Waiting for z-stats file: $TSV_FILE"
    while [ ! -f "$TSV_FILE" ]; do
        sleep 2
    done
    echo "File found! Starting monitoring..."
    sleep 1
fi

echo "Monitoring z-gate statistics from: $TSV_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Function to display latest stats
show_latest() {
    # Get the latest stats for each layer and type
    tail -n 20 "$TSV_FILE" | awk -F'\t' '
    BEGIN {
        print "====== Z-Gate Statistics (Latest) ======"
        print ""
    }
    NR == 1 { next }  # Skip header if present
    {
        step = $1
        layer = $3
        type = $4
        mean = $5
        p10 = $8
        p50 = $9
        p90 = $10
        
        # Store latest for each layer+type combo
        key = layer "_" type
        data[key] = sprintf("L%d %-7s: mean=%6.3f [p10=%5.3f, p50=%5.3f, p90=%5.3f]", 
                           layer, type, mean, p10, p50, p90)
        latest_step = step
    }
    END {
        printf "Step: %d\n\n", latest_step
        
        # Print in sorted order
        for (i = 0; i < 10; i++) {
            if ((i "_runtime") in data) print data[i "_runtime"]
            if ((i "_input") in data)   print data[i "_input"]
            if ((i "_runtime") in data) print ""  # Space between layers
        }
        
        print "----------------------------------------"
        
        # Health check for layer 0 runtime
        if (("0_runtime") in data) {
            # Extract mean from the data string
            match(data["0_runtime"], /mean=([0-9.]+)/, arr)
            mean_val = arr[1] + 0  # Force numeric conversion
            
            if (mean_val < 0.10) {
                print "⚠️  WARNING: Layer 0 very closed (mean = " mean_val " < 0.10)"
                print "   Consider: increase LR or reduce z-bias magnitude"
            } else if (mean_val > 0.50) {
                print "⚠️  WARNING: Layer 0 very open (mean = " mean_val " > 0.50)"
                print "   Consider: decrease LR or increase z-bias magnitude"
            } else if (mean_val >= 0.15 && mean_val <= 0.35) {
                print "✓  Layer 0 z-gate healthy (mean = " mean_val " in [0.15, 0.35])"
            } else if (mean_val >= 0.10 && mean_val < 0.15) {
                print "ℹ️  Layer 0 z-gate slightly closed (mean = " mean_val " in [0.10, 0.15))"
                print "   Monitor for potential plateau"
            } else if (mean_val > 0.35 && mean_val <= 0.50) {
                print "ℹ️  Layer 0 z-gate slightly open (mean = " mean_val " in (0.35, 0.50])"
                print "   Good for fast learning, watch for instability"
            }
        }
    }'
}

# Monitor loop
while true; do
    clear
    show_latest
    sleep 5  # Update every 5 seconds
done
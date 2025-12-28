#!/bin/bash
set -e

# =============================================================================
# ELMAN LADDER OVERNIGHT COMPARISON
# =============================================================================
# Runs all 8 ladder levels (4 normal + 4 log-space) sequentially
# Expected runtime: ~8 hours (1 hour per run on 8 GPUs)
#
# Levels tested:
#   0: Stock Elman (basic tanh)
#   1: Gated Elman (+ delta gate)
#   2: Selective Elman (+ compete×silu)
#   3: Diagonal Selective (diagonal R, like Mamba2)
#
# Each level tested in:
#   - Normal space (standard hidden state)
#   - Log-space (hidden state stored as log magnitude + sign)
# =============================================================================

LOG_DIR="/home/erikg/gruboros/logs"
mkdir -p "$LOG_DIR"

START_TIME=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/overnight_${START_TIME}.log"

echo "=========================================" | tee "$MASTER_LOG"
echo "ELMAN LADDER OVERNIGHT COMPARISON" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"

cd /home/erikg/gruboros

# Track results
declare -A RESULTS

run_experiment() {
    local script=$1
    local name=$2
    local log_file="$LOG_DIR/${name}_${START_TIME}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "=========================================" | tee -a "$MASTER_LOG"
    echo "Starting: $name" | tee -a "$MASTER_LOG"
    echo "Time: $(date)" | tee -a "$MASTER_LOG"
    echo "Log: $log_file" | tee -a "$MASTER_LOG"
    echo "=========================================" | tee -a "$MASTER_LOG"

    if bash "$script" 2>&1 | tee "$log_file"; then
        RESULTS[$name]="SUCCESS"
        echo "✓ $name completed successfully" | tee -a "$MASTER_LOG"
    else
        RESULTS[$name]="FAILED"
        echo "✗ $name FAILED" | tee -a "$MASTER_LOG"
    fi

    # Extract final loss from log if available
    if grep -q "Step.*loss" "$log_file"; then
        FINAL_LOSS=$(grep "Step.*loss" "$log_file" | tail -1 | grep -oP 'loss[= ]+\K[0-9.]+' || echo "N/A")
        echo "Final loss: $FINAL_LOSS" | tee -a "$MASTER_LOG"
        RESULTS[$name]="${RESULTS[$name]} (loss: $FINAL_LOSS)"
    fi

    echo "" | tee -a "$MASTER_LOG"
}

# Run all 8 experiments
echo "" | tee -a "$MASTER_LOG"
echo "Running 8 ladder level comparisons..." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Level 0: Stock Elman
run_experiment "train.ladder_level0.sh" "level0_normal"
run_experiment "train.ladder_level0_log.sh" "level0_logspace"

# Level 1: Gated Elman
run_experiment "train.ladder_level1.sh" "level1_normal"
run_experiment "train.ladder_level1_log.sh" "level1_logspace"

# Level 2: Selective Elman
run_experiment "train.ladder_level2.sh" "level2_normal"
run_experiment "train.ladder_level2_log.sh" "level2_logspace"

# Level 3: Diagonal Selective
run_experiment "train.ladder_level3.sh" "level3_normal"
run_experiment "train.ladder_level3_log.sh" "level3_logspace"

# Print summary
echo "" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"
echo "OVERNIGHT RUN COMPLETE" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "RESULTS SUMMARY:" | tee -a "$MASTER_LOG"
echo "-----------------------------------------" | tee -a "$MASTER_LOG"

for name in "level0_normal" "level0_logspace" "level1_normal" "level1_logspace" \
            "level2_normal" "level2_logspace" "level3_normal" "level3_logspace"; do
    echo "$name: ${RESULTS[$name]:-NOT RUN}" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "Full logs in: $LOG_DIR" | tee -a "$MASTER_LOG"
echo "Output dirs in: /mnt/nvme2n1/erikg/minlms/" | tee -a "$MASTER_LOG"

# Create summary file with output directories
echo "" | tee -a "$MASTER_LOG"
echo "Output directories created:" | tee -a "$MASTER_LOG"
ls -ltd /mnt/nvme2n1/erikg/minlms/*ladder* 2>/dev/null | head -8 | tee -a "$MASTER_LOG" || true

echo "" | tee -a "$MASTER_LOG"
echo "Done!" | tee -a "$MASTER_LOG"

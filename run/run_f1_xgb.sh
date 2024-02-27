#!/usr/bin/env bash

# Set repeats and GPU IDs
REPEAT=5
DATASET="rel-f1"
TASKS=("rel-f1-position")

#"rel-f1-dnf"  "rel-f1-qualifying" 
# Loop over tasks and run experiments
for TASK in "${TASKS[@]}"; do
    mkdir -p "results/xgb_${TASK}"
    python examples/xgboost_baseline.py --repeat $REPEAT --dataset $DATASET --task $TASK > "results/xgb_${TASK}/output.log" 2>&1
done
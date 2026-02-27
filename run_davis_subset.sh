#!/bin/bash

# Configuration
DATA_ROOT="/mnt/hdd/davis_subset"
OUTPUT_DIR="demo_tmp/davis_car_roundabout_multiple_of_14"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over each sequence in the data root
for seq_path in "$DATA_ROOT"/*; do
    # Check if it's a directory
    if [ -d "$seq_path" ]; then
        sequence=$(basename "$seq_path")
        echo "------------------------------------------------"
        echo "Processing sequence: $sequence"
        echo "------------------------------------------------"
        if [ "$sequence" != "car-roundabout" ]; then
            continue
        fi # skip all other sequences
        python demo.py \
            --input "$seq_path" \
            --output_dir "$OUTPUT_DIR" \
            --seq_name "$sequence" \
            --window_wise \
            --window_size 30 \
            --window_overlap_ratio 0.5
    fi
done

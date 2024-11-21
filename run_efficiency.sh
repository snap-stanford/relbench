#!/bin/bash

# Declare the specific range of values
values=(10 100 1000 10000)

# Iterate over the values and run the Python script for each
for i in "${values[@]}"; do
    echo "Running with num_neg_dst_nodes=${i}"
    python3 examples/efficiency.py --num_workers 4 --num_neg_dst_nodes "${i}"
done

echo "All runs completed."
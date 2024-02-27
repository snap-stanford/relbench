#!/usr/bin/env bash

# Set repeats and GPU IDs
REPEATS=5
GPU_IDS="0,1,2,3,4,5"
SLEEP=300

# Test for running a single experiment. --repeat means run how many different random seeds.
python run/sweep.py --config-file run/configs/amazon-churn-sjr.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS --sleep-time $SLEEP
#python run/sweep.py --config-file run/configs/amazon-ltv-sj.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS --sleep-time $SLEEP

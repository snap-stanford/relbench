#!/usr/bin/env bash

# Set repeats and GPU IDs
REPEATS=10
GPU_IDS="0,1,2,4,5,6,7"
SLEEP=60

# Test for running a single experiment. --repeat means run how many different random seeds.
#python run/sweep.py --config-file run/configs/f1-qualifying-sjr.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS --sleep-time $SLEEP
#python run/sweep.py --config-file run/configs/f1-position-sj-sim-sup.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS --sleep-time $SLEEP
python run/sweep.py --config-file run/configs/f1-dnf-sjr.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS  --sleep-time $SLEEP
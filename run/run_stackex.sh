#!/usr/bin/env bash

# Set repeats and GPU IDs
REPEATS=5
GPU_IDS="0,1,2,4,5,6,7"

# Test for running a single experiment. --repeat means run how many different random seeds.
python run/sweep.py --config-file run/configs/stackex-engage-sjr.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS --sleep-time 120
#python run/sweep.py --config-file run/configs/stackex-votes-sj-sim-sup.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS  --sleep-time 120
#python run/sweep.py --config-file run/configs/stackex-badges-sjr.yaml --repeats $REPEATS  --gpu-ids $GPU_IDS  --sleep-time 120
#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python run/main.py --cfg run/configs/example_test.yaml --repeat 1 --auto_select_device

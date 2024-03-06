#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
#CUDA_LAUNCH_BLOCKING=1 python run/main.py --cfg run/configs/example_test.yaml --repeat 1 --auto_select_device
python run/main.py --cfg run/configs/example_test.yaml --repeat 1 --auto_select_device
#python run/main.py --cfg results/f1-dnf_model.channels_64_128_model.use_self_join_True_ optim.base_lr_0.01_0.001_selfjoin.sim_score_type_None_L2_attention/f1-dnf_model.channels_128_model.use_self_join_True_optim.base_lr_0.01_selfjoin.sim_score_type_attention_run.yaml --repeat 3 --auto_select_device

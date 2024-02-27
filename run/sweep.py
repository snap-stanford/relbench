import argparse
import itertools
import os
import subprocess
import time
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, Generator, Iterable, List, Tuple

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=False, default="run/configs/base.yaml"
    )
    parser.add_argument("--gpu-ids", type=str, required=False, default="0,1,2,3")
    parser.add_argument("--repeats", type=int, required=False, default=1)
    parser.add_argument("--sleep-time", type=int, required=False, default=30)
    args = parser.parse_args()

    fast_sweep = False

    # Define the hyperparameters and their values to sweep over
    hyperparameters = {
        "model.channels": [128, 256],
        # 'model.conv': ['sage', 'gat'],
        "model.num_layers": [2],
        "model.use_self_join_with_retrieval": [True, False],
        # 'model.aggr': ['sum', 'mean', 'max'],
        "model.dropout": [0.0],
        "model.feature_dropout": [0.0, 0.2],
        "selfjoin.memory_bank_size": [2048],
        #'selfjoin.num_filtered': [32, 128, 512],
        # 'model.hetero_aggr': ['sum', 'mean', 'max'],
        "optim.base_lr": [0.01, 0.001],
        # 'loader.num_neighbors': [16, 32, 64, 128, 256],
        # 'loader.temporal_strategy': ['uniform', 'last']
        # 'selfjoin.node_type_considered': ['drivers', None],
        # 'selfjoin.num_filtered': [10, 20, 50],
        #'selfjoin.sim_score_type': [None, 'cos', 'L2', 'attention'],
        # 'selfjoin.aggr_scheme': ['gat', 'mpnn'],
        # 'selfjoin.normalize_score': [True, False],
    }

    repeats = args.repeats  # number of seeds to run
    gpu_ids = args.gpu_ids.split(",")  # Define the GPU IDs available
    original_config_file = (
        args.config_file
    )  # Specify the path to the original config file

    # Create a folder to store the YAML files
    output_folder = "results/" + original_config_file.split("/")[-1][:-5]
    original_config_name = output_folder.split("/")[-1]

    for param, values in hyperparameters.items():
        value_str = "_".join(str(value) for value in values)
        output_folder += f"_{param}_{value_str}"

    print(f"config files output to {output_folder}")
    os.makedirs(output_folder, exist_ok=False)

    # Generate all combinations of hyperparameter values
    combinations = list(itertools.product(*hyperparameters.values()))

    with open(original_config_file, "r") as template_config:
        template = yaml.safe_load(template_config)

    # Maintain resource pool of available GPUs.
    resource_pool = Queue()
    for gpu in map(int, gpu_ids):  # provide GPU ids as comma-separated list
        resource_pool.put(gpu)

    def create_worker(
        new_config_file: str, gpu: int, exp_id: int
    ) -> Callable[[], None]:
        def worker() -> None:
            print(
                f"Started: Exp {exp_id} Config {new_config_file.split('/')[-1]}, GPU {gpu}."
            )
            command = f"CUDA_VISIBLE_DEVICES={gpu} python run/main.py --cfg {new_config_file} --repeat {repeats} > /dev/null 2>&1"
            # command = f'CUDA_VISIBLE_DEVICES={gpu} python run/main.py --cfg {new_config_file} --repeat {repeats}'
            # print(command)
            subprocess.run(command, shell=True)
            print(
                f"Finished: Exp {exp_id} Config {new_config_file.split('/')[-1]}, GPU {gpu}."
            )
            resource_pool.put(gpu)

        return worker

    # Iterate over hyperparameter combinations and GPU IDs
    for combo in combinations:
        # Create a new config dictionary based on the template
        new_config = template.copy()
        new_config["out_dir"] = output_folder

        for param, value in zip(hyperparameters.keys(), combo):
            param_parts = param.split(".")
            update_nested_dict(new_config, param_parts, value)

        # Write the updated config to a YAML file
        new_config_file = config_name(
            combo, hyperparameters, output_folder, original_config_name
        )

        with open(new_config_file, "w") as config_file:
            yaml.dump(new_config, config_file, default_flow_style=False)

    # Run each hyperparameter configuration.
    for exp_id, combo in enumerate(combinations):
        new_config_file = config_name(
            combo, hyperparameters, output_folder, original_config_name
        )
        gpu = resource_pool.get()  # wait for a GPU to become available
        worker = Process(target=create_worker(new_config_file, gpu, exp_id))
        worker.start()

        # Wait for a while to avoid launching jobs too quickly
        time.sleep(args.sleep_time)


def update_nested_dict(d, key_list, value):
    if len(key_list) == 1:
        if key_list[0] in d:
            d[key_list[0]] = value
        else:
            raise KeyError(f"Key not found: {key_list[0]}")
    elif key_list[0] in d:
        update_nested_dict(d[key_list[0]], key_list[1:], value)
    else:
        raise KeyError(f"Key not found: {key_list[0]}")


def config_name(combo, hyperparameters, output_folder, original_config_name):
    config_name = "_".join(
        f"{param}_{value}" for param, value in zip(hyperparameters.keys(), combo)
    )
    config_name = "_".join([original_config_name, config_name])
    new_config_file = os.path.join(output_folder, config_name + "_run.yaml")

    return new_config_file


if __name__ == "__main__":
    main()

# python run/sweep.py --config-file run/configs/stackex-engage.yaml --gpu-ids 0,1,2,3,4,5,6,7,8,9 --repeats 1

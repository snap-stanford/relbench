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
    parser.add_argument("--gpu-ids", type=str, required=False, default="4,5,6,7")
    parser.add_argument("--repeats", type=int, required=False, default=5)
    parser.add_argument("--sleep-time", type=int, required=False, default=30)
    args = parser.parse_args()

    gpu_ids = args.gpu_ids.split(",")  # Define the GPU IDs available

    """
    grid1 = {"DATASET": ["rel-f1"],
             "TASK": ["rel-f1-dnf", "rel-f1-qualifying", "rel-f1-position"], 

             "MODEL": ["lgbm", "baseline"]}

    grid2 = {"DATASET": ["rel-stackex"],
             "TASK": ["rel-stackex-engage", "rel-stackex-votes", "rel-stackex-badges"],
             "MODEL": ["lgbm"]}
    """
    grid3 = {"DATASET": ["rel-trial"],
             "TASK": ["rel-trial-site"],
             "MODEL": ["lgbm"]}

    grids = [grid3]

    combinations = []
    for grid in grids:

        os.makedirs(
            os.path.join("results", "baselines", grid["DATASET"][0]), exist_ok=True
        )


        assert list(grid.keys()) == ["DATASET", "TASK", "MODEL"]

        # Use list concatenation to directly add combinations to the list
        combinations += list(itertools.product(*grid.values()))

    # Print or use combinations as needed
    print(combinations)

    def get_launch_command(MODEL):
        if MODEL == "lgbm":
            return "python examples/lightgbm_baseline.py"
        elif MODEL == "baseline":
            return "python examples/baseline.py"
        
    def get_log_file(DATASET, TASK, MODEL):
        return f"results/baselines/{DATASET}/{TASK}_{MODEL}.log"
    
    # Maintain resource pool of available GPUs.
    resource_pool = Queue()
    for gpu in map(int, gpu_ids):  # provide GPU ids as comma-separated list
        resource_pool.put(gpu)

    def create_worker(
        DATASET: str,
        TASK: str,
        MODEL: str,
        gpu: int,
    ) -> Callable[[], None]:
        def worker() -> None:
            print(f"Started: Exp {DATASET} {TASK} {MODEL}")
            command = f"CUDA_VISIBLE_DEVICES={gpu} {get_launch_command(MODEL)} --dataset {DATASET} --task {TASK} --repeats {args.repeats}"
            log_file = get_log_file(DATASET, TASK, MODEL)
            with open(log_file, "w") as f:
                subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

            resource_pool.put(gpu)

        return worker


    # Run each hyperparameter configuration.
    for combo in combinations:
        gpu = resource_pool.get()  # wait for a GPU to become available
        worker = Process(target=create_worker(*combo, gpu=gpu))
        worker.start()

        # Wait for a while to avoid launching jobs too quickly
        time.sleep(args.sleep_time)



if __name__ == "__main__":
    main()

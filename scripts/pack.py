import json
import tempfile
import time

import relbench
from relbench.datasets import get_dataset

registry = relbench._pooch.registry
print(f"old registry = {json.dumps(registry, indent=2)}")

delta = {}

tmpdir = tempfile.mkdtemp()
print(f"{tmpdir = }")

for dataset_name, task_names in [
    # ("rel-amazon", ["rel-amazon-churn", "rel-amazon-ltv"]),
    # ("rel-stackex", ["rel-stackex-engage", "rel-stackex-votes"]),
    ("rel-math-stackex", []),
    ("rel-f1", []),
    # ("rel-trial", []),
    ("rel-hm", []),
]:
    print(f"{dataset_name = }")

    print(f"make db...")
    tic = time.time()
    ## for quick testing
    # dataset = get_dataset(
    #     dataset_name, category="fashion", use_5_core=True, process=True
    # )
    dataset = get_dataset(dataset_name, process=True)
    toc = time.time()
    print(f"took {toc - tic:.2f} s.")

    print(f"pack db...")
    tic = time.time()
    fname, sha256 = dataset.pack_db(tmpdir)
    delta[fname] = sha256
    toc = time.time()
    print(f"took {toc - tic:.2f} s.")

    for task_name in task_names:
        print(f"{task_name = }")

        print(f"make and pack tables...")
        tic = time.time()
        task = dataset.get_task(task_name, process=True)
        fname, sha256 = task.pack_tables(tmpdir)
        delta[fname] = sha256
        toc = time.time()
        print(f"took {toc - tic:.2f} s.")

    print(f"===")
    print(f"delta = {json.dumps(delta, indent=2)}")

    registry.update(delta)
    print(f"new registry = {json.dumps(registry, indent=2)}")

    print(f"scp -r {tmpdir}/* relbench.stanford.edu:/lfs/0/relbench/staging_data")
    print(f"===")

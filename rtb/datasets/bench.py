import os
import shutil
import time
from pathlib import Path
from typing import Dict, NamedTuple, Union

import pandas as pd

from rtb.data import Database, Dataset, Task
from rtb.tasks.product import ChurnTask, LTVTask
from rtb.utils import download_url

url_fmt = "http://rtb.stanford.edu/data/{}.zip"


class DatasetInfo(NamedTuple):
    train_max_time: pd.Timestamp
    val_max_time: pd.Timestamp
    task_cls_dict: Dict[str, type[Task]]


dataset_dict = {
    "product": DatasetInfo(
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2017-01-01"),
        {
            "churn": ChurnTask,
            "ltv": LTVTask,
        },
    ),
}


def get_dataset(name: str, root=Union[str, os.PathLike], download=False) -> Dataset:
    """

    Conventions:

    url points to <name>.zip
    <name>.zip is downloaded to <root>/<name>.zip
    unzipping <root>/<name>.zip creates <root>/<name>/
    <root>/<name> can be loaded with Database.load
    """

    url = url_fmt.format(name)

    if download or not Path(f"{root}/{name}").exists():
        print(f"downloading from {url} to {root}...")
        tic = time.time()
        path = download_url(url, root)
        toc = time.time()
        print(f"downloaded in {toc - tic:.2f} s.")

        print(f"extracting {path} to {root}...")
        tic = time.time()
        shutil.unpack_archive(path, root)
        toc = time.time()
        print(f"extracted in {toc - tic:.2f} s.")

    else:
        print(f"{root}/{name} exists, skipping download.")

    dataset_info = dataset_dict[name]

    super().__init__(
        db=Database.load(f"{root}/{name}"),
        train_max_time=dataset_info.train_max_time,
        val_max_time=dataset_info.val_max_time,
        tasks=dataset_info.task_cls_dict,
    )

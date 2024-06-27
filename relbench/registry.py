from collections import defaultdict
from functools import lru_cache

import pooch

from relbench.data import Dataset, BaseTask

download_registry = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry={
        "rel-amazon/db.zip": "2fb5d1b6f0d8886374bc25b3a81becbe191ad30e524ac7fb998bab4c87010adc",
        "rel-avito/db.zip": "09fe913ece4f17f79ca0d2c1d25ed9f6f7e803fa4a08dcf520b7a0e73f34b1ed",
        "rel-event/db.zip": "141f4842600d091250c1f94e4c479c35e76d7ec3aef9155316f83d4828d85e5e",
        "rel-f1/db.zip": "e41ca0d69d54f16b408fe03b6c19b772ae701336cf84260ef5c84fca798a1422",
        "rel-hm/db.zip": "6ff6537f2fed885c5c8a94525364678dea206c57006de0edb4d76ca71c9c114e",
        "rel-stack/db.zip": "b703d141f86c210e9e6809807ec0bcf9b3e2fcd32a679835c9d71d8048b89188",
        "rel-trial/db.zip": "76093dae4365839cae4f949cc2c982c8c8ddf9886e309d84606b37208c8102da",
    },
)

dataset_registry = {}
task_registry = defaultdict(dict)


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
):
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names():
    return list(dataset_registry.keys())


@lru_cache(maxsize=None)
def get_dataset(name: str) -> Dataset:
    cls, args, kwargs = dataset_registry[name]
    dataset = cls(*args, **kwargs)
    return dataset


def register_task(
    dataset_name: str,
    task_name: str,
    cls,
    *args,
    **kwargs,
):
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str):
    return list(task_registry[dataset_name].keys())


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str) -> BaseTask:
    dataset = get_dataset(dataset_name)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task

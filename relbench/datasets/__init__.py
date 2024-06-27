from functools import lru_cache

from ..data import Dataset
from . import amazon, avito, event, f1, hm, stack, trial

dataset_registry = {}


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


register_dataset("rel-amazon", amazon.AmazonDataset)
register_dataset("rel-avito", avito.AvitoDataset)
register_dataset("rel-event", event.EventDataset)
register_dataset("rel-f1", f1.F1Dataset)
register_dataset("rel-hm", hm.HMDataset)
register_dataset("rel-stack", stack.StackDataset)
register_dataset("rel-trial", trial.TrialDataset)

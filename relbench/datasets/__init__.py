from relbench.data import RelBenchDataset
from relbench.datasets.amazon import AmazonDataset
from relbench.datasets.fake import FakeDataset
from relbench.datasets.stackex import StackExDataset
from relbench.datasets.f1 import F1Dataset

dataset_cls_list = [
    AmazonDataset,
    StackExDataset,
    F1Dataset,
    FakeDataset,
]

dataset_cls_dict = {dataset_cls.name: dataset_cls for dataset_cls in dataset_cls_list}

dataset_names = list(dataset_cls_dict.keys())


def get_dataset(name: str, *args, **kwargs) -> RelBenchDataset:
    r"""Returns a dataset by name."""
    return dataset_cls_dict[name](*args, **kwargs)


__all__ = [
    "AmazonDataset",
    "StackExDataset",
    "F1Dataset",
    "FakeDataset",
    "dataset_cls_dict",
    "dataset_names",
    "get_dataset",
]

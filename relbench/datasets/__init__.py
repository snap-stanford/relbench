from relbench.datasets.amazon import AmazonDataset
from relbench.datasets.fake import FakeDataset
from relbench.datasets.stackex import StackExDataset

dataset_cls_list = [
    AmazonDataset,
    StackExDataset,
    FakeDataset,
]

dataset_cls_dict = {dataset_cls.name: dataset_cls for dataset_cls in dataset_cls_list}

dataset_names = list(dataset_cls_dict.keys())


def get_dataset(name: str, *args, **kwargs) -> "Dataset":
    r"""Returns a dataset by name."""
    return dataset_cls_dict[name](*args, **kwargs)


__all__ = [
    "AmazonDataset",
    "StackExDataset",
    "FakeDataset",
    "dataset_cls_dict",
    "dataset_names",
    "get_dataset",
]

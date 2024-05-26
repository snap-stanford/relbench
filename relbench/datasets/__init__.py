from relbench.data import RelBenchDataset
from relbench.datasets.amazon import AmazonDataset
from relbench.datasets.f1 import F1Dataset
from relbench.datasets.fake import FakeDataset
from relbench.datasets.hm import HMDataset
from relbench.datasets.math_stackex import MathStackExDataset
from relbench.datasets.stackex import StackExDataset
from relbench.datasets.trial import TrialDataset

dataset_cls_list = [
    AmazonDataset,
    StackExDataset,
    MathStackExDataset,
    F1Dataset,
    TrialDataset,
    FakeDataset,
    HMDataset,
]

dataset_cls_dict = {dataset_cls.name: dataset_cls for dataset_cls in dataset_cls_list}

dataset_names = list(dataset_cls_dict.keys())


def get_dataset(name: str, *args, **kwargs) -> RelBenchDataset:
    r"""Returns a dataset by name."""
    return dataset_cls_dict[name](*args, **kwargs)


def decompress_gz_file(input_path : str, output_path : str):
    import gzip
    import shutil
    # Open the gz file in binary read mode
    with gzip.open(input_path, 'rb') as f_in:
        # Open the output file in binary write mode
        with open(output_path, 'wb') as f_out:
            # Copy the decompressed data from the gz file to the output file
            shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed file saved as: {output_path}")

__all__ = [
    "AmazonDataset",
    "StackExDataset",
    "MathStackExDataset",
    "F1Dataset",
    "TrialDataset",
    "FakeDataset",
    "dataset_cls_dict",
    "dataset_names",
    "get_dataset",
]

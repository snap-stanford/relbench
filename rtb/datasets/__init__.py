from rtb.datasets.amazon_reviews import AmazonReviewsDataset
from rtb.datasets.fake_reviews import FakeReviewsDataset

dataset_classes = [
    AmazonReviewsDataset,
    FakeReviewsDataset,
]

dataset_dict = {cls.__name__: cls for cls in dataset_classes}


def get_dataset(name: str) -> "Dataset":
    r"""Returns a dataset by name."""

    return dataset_dict[name]()


__all__ = [
    "AmazonReviewsDataset",
    "FakeReviewsDataset",
]

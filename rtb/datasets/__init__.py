from rtb.datasets.fake import FakeProductDataset
from rtb.datasets.forum import ForumDataset
from rtb.datasets.grant import GrantDataset
from rtb.datasets.product import ProductDataset


def get_dataset(name: str, *args, **kwargs):
    r"""Convenience function to get a dataset by name."""

    if name == FakeProductDataset.name:
        return FakeProductDataset(*args, **kwargs)
    if name == ProductDataset.name:
        return ProductDataset(*args, **kwargs)
    if name == GrantDataset.name:
        return GrantDataset(*args, **kwargs)
    if name == ForumDataset.name:
        return ForumDataset(*args, **kwargs)

    raise ValueError(f"Unknown dataset name: '{name}'")


__all__ = [
    "FakeProductDataset",
    "ForumDataset",
    "GrantDataset",
    "ProductDataset",
    "get_dataset",
]

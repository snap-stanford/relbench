from rtb.datasets.forum import ForumDataset
from rtb.datasets.grant import GrantDataset
from rtb.datasets.product import ProductDataset


def get_dataset(name: str, *args, **kwargs):
    r"""Convenience function to get a dataset by name."""

    if name == "rtb-product":
        return ProductDataset(*args, **kwargs)
    if name == "rtb-grant":
        return GrantDataset(*args, **kwargs)
    if name == "rtb-forum":
        return ForumDataset(*args, **kwargs)

    raise ValueError(f"Unknown dataset name: '{name}'")


__all__ = [
    "ProductDataset",
    "GrantDataset",
    "ForumDataset",
    "get_dataset",
]

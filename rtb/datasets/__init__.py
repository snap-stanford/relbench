from rtb.datasets.product import ProductDataset


def get_dataset(name: str, *args, **kwargs):
    r"""Convenience function to get a dataset by name."""

    if name == "rtb-product":
        return ProductDataset(*args, **kwargs)

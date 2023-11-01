import rtb


def get_dataset(name: str, *args, **kwargs):
    r"""Convenience function to get a dataset by name."""

    if name == "mtb-product":
        return rtb.datasets.ProductDataset(*args, **kwargs)

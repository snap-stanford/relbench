import numpy as np
import pandas as pd
import torch
from torch import Tensor


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = torch.from_numpy(ser.astype(int).values)
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time

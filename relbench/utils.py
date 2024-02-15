import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pooch
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


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    shutil.unpack_archive(zip_path, unzip_path)
    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
    ]

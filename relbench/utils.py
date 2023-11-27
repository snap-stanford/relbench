import os
import shutil
from pathlib import Path
from typing import Union

import pandas as pd
import pooch
import requests
from tqdm import tqdm


def to_unix_time(column: pd.Series) -> pd.Series:
    """convert a timestamp column to unix time"""
    # return pd.to_datetime(column).astype('int64') // 10**9
    return pd.to_datetime(column).astype("datetime64[s]")


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    shutil.unpack_archive(zip_path, unzip_path)
    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
    ]

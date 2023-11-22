import os
import shutil
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from tqdm import tqdm


def rolling_window_sampler(
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
    window_size: pd.Timedelta,
    stride: pd.Timedelta,
) -> Tuple[pd.Series[pd.Timestamp], pd.Series[pd.Timestamp]]:
    # Traverse backwards to operate on the latest available timestamps:
    # TODO: verify closedness
    window_max_time = pd.date_range(
        end_time, start_time + window_size, freq=stride, closed="right"
    )
    window_min_time = window_max_time - window_size
    return window_min_time, window_max_time


def one_window_sampler(
    min_time: pd.Timestamp, window_size: pd.Timestamp
) -> Tuple[pd.Series[pd.Timestamp], pd.Series[pd.Timestamp]]:
    return pd.Series([min_time]), pd.Series([min_time + window_size])


def to_unix_time(column: pd.Series) -> pd.Series:
    """convert a timestamp column to unix time"""
    # return pd.to_datetime(column).astype('int64') // 10**9
    return pd.to_datetime(column).astype("datetime64[s]")


def download_url(
    url: str,
    root: str,
) -> str:
    r"""Downloads the content of :obj:`url` to the specified folder
    :obj:`root`.

    Args:
        url (str): The URL.
        root (str): The root folder.
    """

    filename = url.rpartition("/")[2]
    path = os.path.join(root, filename)
    if os.path.exists(path):
        return path

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    return path


def download_and_extract(url: str, root: Union[str, os.PathLike]) -> None:
    download_path = download_url(url, root)
    shutil.unpack_archive(download_path, root)

import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm


def rolling_window_sampler(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    window_size: pd.Timedelta,
    stride: pd.Timedelta,
) -> pd.DataFrame:
    """Returns a DataFrame with columns window_min_time and window_max_time."""

    df = pd.DataFrame()
    start_time = int(start_time.timestamp())
    end_time = int(end_time.timestamp())
    window_size = int(window_size.total_seconds())
    stride = int(stride.total_seconds())

    df["window_min_time"] = range(
        # TODO: find a better way to do this
        # start_time should be excluded, plus 1 second
        start_time + 1,
        end_time - window_size,  # window should not overshoot end_time
        stride,
    )
    df["window_max_time"] = df["window_min_time"] + window_size
    df["window_min_time"] = df["window_min_time"].astype("datetime64[s]")
    df["window_max_time"] = df["window_max_time"].astype("datetime64[s]")
    return df


def one_window_sampler(
    start_time: pd.Timestamp, window_size: pd.Timestamp
) -> pd.DataFrame:
    """Returns a DataFrame with columns window_min_time and window_max_time."""
    start_time = int(start_time.timestamp())
    window_size = int(window_size.total_seconds())
    df = pd.DataFrame()
    # TODO: find a better way to do this
    df["window_min_time"] = [start_time + 1]  # plus 1 second
    df["window_max_time"] = [start_time + window_size]
    df["window_min_time"] = df["window_min_time"].astype("datetime64[s]")
    df["window_max_time"] = df["window_max_time"].astype("datetime64[s]")
    return df


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


def unzip(path, root):
    r"""
    Args:
        path (str): The path to the zip file that needs to be extracted.
        root (str): The directory where the contents of the zip file will be extracted.
    """
    with zipfile.ZipFile(path, "r") as zip:
        zip.extractall(path=root)

import os
import zipfile
from typing import List, Union

import pandas as pd
import requests
import torch
import torch_frame as pyf
import torch_geometric as pyg
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm

from rtb.data.database import Database
from rtb.data.table import Table


def to_pyf_dataset(table: Table) -> pyf.data.Dataset:
    r"""Converts a Table to a PyF Dataset.

    Primary key and foreign keys are removed in this process."""

    raise NotImplementedError


def make_pkey_fkey_graph(db: Database) -> pyg.data.HeteroData:
    """
    Models the database as a heterogeneous graph.

    Instead of node embeddings in data.x, we store the tensor frames in data.tf.
    """
    data = pyg.data.HeteroData()

    for table_name, table in db.tables.items():
        # Materialize the tables:
        # TODO Convert to PyTorch Frame (needs stype information though)
        data[table_name].df = table.df

        # Add time attribute:
        if table.time_col is not None:
            time = table.df[table.time_col].values
            data[table_name].time = torch.from_numpy(time)

        # Add edges:
        for fkey_name, dst_table_name in table.fkeys.items():
            dst_table = db.tables[dst_table_name]

            # TODO Needs standardized integer-based links.
            fkey_idx = torch.from_numpy(table.df[fkey_name].values)
            pkey_idx = torch.from_numpy(dst_table.df[dst_table.pkey].values)

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_idx, pkey_idx], dim=0)
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            edge_type = (table_name, f"f2p_{fkey_name}", dst_table_name)
            data[edge_type].edge_index = edge_index

            # pkey -> fkey edges
            edge_index = torch.stack([pkey_idx, fkey_idx], dim=0)
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            edge_type = (dst_table_name, f"p2f_{fkey_name}", table_name)
            data[edge_type].edge_index = edge_index

    return data


class AddTargetLabelTransform:
    r"""Adds the target label to the batch. The batch consists of disjoint
    subgraphs loaded via temporal sampling. The same input node can occur twice
    with different timestamps, and thus different subgraphs and labels. Hence
    labels cannot be stored in the Data object directly, and must be attached
    to the batch after the batch is created."""

    def __init__(self, labels: List[Union[int, float]]):
        self.labels = torch.tensor(labels)

    def __call__(self, batch: pyg.data.Batch) -> pyg.data.Batch:
        batch.y = self.labels[batch.input_id]
        return batch


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
        # start_time should be excluded, plus 1 second
        start_time + 1,
        end_time - window_size,
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

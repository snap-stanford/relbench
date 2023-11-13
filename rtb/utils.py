from typing import List, Union

import pandas as pd
import torch_frame as pyf
import torch_geometric as pyg

from rtb.data.table import Table
from rtb.data.database import Database


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

    for name, table in db.tables.items():
        # materialize the tables
        pyf_dataset = to_pyf_dataset(table)
        pyf_dataset.materialize()
        data[name].tf = pyf_dataset.tensor_frame

        # add time attribute
        data[name].time_stamp = torch.tensor(table.df[table.time_col])

        # add edges
        for col_name, pkey_name in table.fkeys.items():
            fkey_idx = torch.tensor(table.df[table.primary_key])
            pkey_idx = torch.tensor(table.df[col_name])

            # fkey -> pkey edges
            data[name, "f2p::" + col_name, pkey_name].edge_index = torch.stack(
                [fkey_idx, pkey_idx]
            )
            # pkey -> fkey edges
            data[pkey_name, "p2f::" + col_name, name].edge_index = torch.stack(
                [pkey_idx, fkey_idx]
            )

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
    df["window_min_time"] = [start_time + 1] ## plus 1 second
    df["window_max_time"] = [start_time + window_size]
    df["window_min_time"] = df["window_min_time"].astype("datetime64[s]")
    df["window_max_time"] = df["window_max_time"].astype("datetime64[s]")
    return df


def to_unix_time(column: pd.Series) -> pd.Series:
    """convert a timestamp column to unix time"""
    # return pd.to_datetime(column).astype('int64') // 10**9
    return pd.to_datetime(column).astype("datetime64[s]")

import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.utils import infer_df_stype
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from rtb.data import Database, Table
from rtb.data.task import Task, TaskType


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    return torch.from_numpy(ser.astype(int).values) // 10**9


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, Any]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): : The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.tables.items():
        # Take the first 10,000 rows for quick stype inference.
        inferred_col_to_stype = infer_df_stype(table.df)

        # Temporarily removing time_col since StypeEncoder for
        # stype.timestamp is not yet supported.
        # TODO: Drop the removing logic once StypeEncoder is supported.
        # https://github.com/pyg-team/pytorch-frame/pull/225
        inferred_col_to_stype = {
            col_name: inferred_stype
            for col_name, inferred_stype in inferred_col_to_stype.items()
            if inferred_stype != stype.timestamp
        }

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        if table.pkey_col is not None:
            inferred_col_to_stype.pop(table.pkey_col)
        for fkey in table.fkey_col_to_pkey_table.keys():
            inferred_col_to_stype.pop(fkey)

        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype

    return inferred_col_to_stype_dict


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> HeteroData:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with
    primary-foreign key relationships, together with the column stats of each
    table.

    Args:
        db (Database): A database object containing a set of tables.
        col_to_stype_dict (Dict[str, Dict[str, stype]]): Column to stype for
            each table.
        cache_dir (str, optional): A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scrach without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature..
    """
    data = HeteroData()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.tables.items():
        # Materialize the tables into tensor frames:
        df = table.df
        col_to_stype = col_to_stype_dict[table_name]

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            df = pd.DataFrame({"__const__": np.ones(len(table.df))})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        data[table_name].col_stats = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = to_unix_time(table.df[table.time_col])

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = table.df[fkey_name]
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))

            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = edge_index

            # pkey -> fkey edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"p2f_{fkey_name}", table_name)
            data[edge_type].edge_index = edge_index

    return data


class AttachTargetTransform:
    r"""Adds the target label to the heterogeneous mini-batch.
    The batch consists of disjoins subgraphs loaded via temporal sampling.
    The same input node can occur twice with different timestamps, and thus
    different subgraphs and labels. Hence labels cannot be stored in the graph
    object directly, and must be attached to the batch after the batch is
    created."""

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class TrainTableInput(NamedTuple):
    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_train_table_input(
    train_table: Table,
    task: Task,
) -> TrainTableInput:
    assert len(train_table.fkey_col_to_pkey_table) == 1
    fkey_col, table_name = list(train_table.fkey_col_to_pkey_table.items())[0]

    nodes = torch.from_numpy(train_table.df[fkey_col].astype(int).values)

    time: Optional[Tensor] = None
    if train_table.time_col is not None:
        time = to_unix_time(train_table.df[train_table.time_col])

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in train_table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        target = torch.from_numpy(
            train_table.df[task.target_col].values.astype(target_type)
        )
        transform = AttachTargetTransform(table_name, target)

    return TrainTableInput(
        nodes=(table_name, nodes),
        time=time,
        target=target,
        transform=transform,
    )

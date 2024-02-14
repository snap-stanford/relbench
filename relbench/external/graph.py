import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
from torch_geometric.utils import sort_edge_index

from relbench.data import Database, LinkTask, NodeTask, Table


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = torch.from_numpy(ser.astype(int).values)
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, Any]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): : The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.table_dict.items():
        inferred_col_to_stype = infer_df_stype(table.df)
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
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        if table.pkey_col is not None:
            col_to_stype.pop(table.pkey_col)
        for fkey in table.fkey_col_to_pkey_table.keys():
            col_to_stype.pop(fkey)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            df = pd.DataFrame({"__const__": np.ones(len(table.df))})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        data[table_name].col_stats = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = to_unix_time(table.df[table.time_col])

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"p2f_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

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


class NodeTrainTableInput(NamedTuple):
    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(
    table: Table,
    task: NodeTask,
) -> NodeTrainTableInput:
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = to_unix_time(table.df[table.time_col])

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == "multiclass_classification":
            target_type = int
        target = torch.from_numpy(table.df[task.target_col].values.astype(target_type))
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


class LinkTrainTableInput(NamedTuple):
    r"""Trainining table input for link prediction.

    - source_nodes is a Tensor of source node indices.
    - destination_nodes[i] gives a list of destination node indices for source_nodes[i]
    - num_destination_nodes is the total number of destination nodes
        (used to perform negative sampling).
    - time is a Tensor of time for source_nodes
    """
    source_nodes: Tuple[NodeType, Tensor]
    destination_nodes_list: Tuple[NodeType, List[List[int]]]
    num_destination_nodes: int
    time: Optional[Tensor]


def get_link_train_table_input(
    table: Table,
    task: LinkTask,
) -> LinkTrainTableInput:
    source_node_idx: Tensor = torch.from_numpy(
        table.df[task.source_entity_col].astype(int).values
    )
    destination_node_idx: List[List[int]] = table.df[
        task.destination_entity_col
    ].to_list()
    num_destination_nodes = len(
        task.dataset.db.table_dict[task.destination_entity_table]
    )

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = to_unix_time(table.df[table.time_col])

    return LinkTrainTableInput(
        source_nodes=(task.source_entity_table, source_node_idx),
        destination_nodes_list=(task.destination_entity_table, destination_node_idx),
        num_destination_nodes=num_destination_nodes,
        time=time,
    )

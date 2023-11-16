from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
import torch
from rtb.data import Database, Table
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset, StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    return torch.from_numpy(ser.astype(int).values) // 10**9


# TODO: fix
def dummy_text_embedder(input: List[str]) -> torch.Tensor:
    r"""Dummy text embedder."""
    return torch.rand(len(input), 768)


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with
    primary-foreign key relationships, together with the column stats of each
    table.

    Args:
        db (Database): A database object containing a set of tables.
        col_to_stype_dict (Dict[str, Dict[str, stype]]): Column to stype for
            each table.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
        Dict[str, Dict[str, Dict[StatType, Any]]]: Column stats dictionary,
            mapping table name into column stats.
    """
    data = HeteroData()

    for table_name, table in db.tables.items():
        # Materialize the tables into tensor frames:
        dataset = Dataset(
            df=table.df,
            col_to_stype=col_to_stype_dict[table_name],
            # TODO: fix
            text_embedder_cfg=TextEmbedderConfig(text_embedder=dummy_text_embedder),
        ).materialize()

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
    target_col: str,
    target_dtype: Optional[torch.dtype] = None,
) -> TrainTableInput:
    assert len(train_table.fkey_col_to_pkey_table) == 1
    fkey_col, table_name = list(train_table.fkey_col_to_pkey_table.items())[0]

    nodes = torch.from_numpy(train_table.df[fkey_col].values)

    time: Optional[Tensor] = None
    if train_table.time_col is not None:
        time = to_unix_time(train_table.df[train_table.time_col])

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if target_col in train_table.df:
        target = torch.from_numpy(train_table.df[target_col].values)
        if target_dtype is not None:
            target = target.to(target_dtype)
        transform = AttachTargetTransform(table_name, target)

    return TrainTableInput(
        nodes=(table_name, nodes),
        time=time,
        target=target,
        transform=transform,
    )

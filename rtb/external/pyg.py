from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from rtb.data import Table
from rtb.data.database import Database
from torch_frame import stype
from torch_frame.data import Dataset, StatType
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import sort_edge_index


def _drop_pkey_fkey(table: Table) -> pd.DataFrame:
    drop_keys = []
    if table.pkey_col is not None:
        drop_keys.append(table.pkey_col)
    drop_keys.extend(list(table.fkeys.keys()))
    return table.df.drop(drop_keys, axis=1)


def _map_index(index_map: pd.Series, ser: pd.Series) -> torch.Tensor:
    return torch.from_numpy(
        pd.merge(
            ser.rename("data"),
            index_map,
            how="left",
            left_on="data",
            right_index=True,
        )["index"].values
    )


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
    # Obtain index mapping of primary keys
    index_map_dict: Dict[str, Tuple[str, pd.Series]] = {}
    for table_name, table in db.tables.items():
        if table.pkey_col is not None:
            table_pkey_col = table.df[table.pkey_col]
            if table_pkey_col.nunique() != len(table_pkey_col):
                raise RuntimeError(
                    f"The primary key '{table_pkey_col}' of "
                    f"table {table_name} contains duplicated elements."
                )
            index_map: pd.Series = pd.Series(
                index=table.df[table.pkey_col],
                data=pd.RangeIndex(0, len(table.df[table.pkey_col])),
                name="index",
            )
            index_map_dict[table_name] = index_map

    col_stats_dict = {}
    for table_name, table in db.tables.items():
        # Materialize the tables:
        dataset = Dataset(
            df=_drop_pkey_fkey(table), col_to_stype=col_to_stype_dict[table_name]
        ).materialize()
        data[table_name] = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            time = table.df[table.time_col].astype(int).values / 10**9
            data[table_name].time = torch.from_numpy(time)

        # Add edges:
        for fkey_name, dst_table_name in table.fkeys.items():
            pkey_idx = _map_index(index_map_dict[dst_table_name], table.df[fkey_name])
            fkey_idx = torch.arange(len(pkey_idx))

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

    return data, col_stats_dict


class AddTargetLabelTransform:
    r"""Adds the target label to the batch. The batch consists of disjoint
    subgraphs loaded via temporal sampling. The same input node can occur twice
    with different timestamps, and thus different subgraphs and labels. Hence
    labels cannot be stored in the Data object directly, and must be attached
    to the batch after the batch is created."""

    def __init__(self, labels: List[Union[int, float]]):
        self.labels = torch.tensor(labels)

    def __call__(self, batch: Batch) -> Batch:
        batch.y = self.labels[batch.input_id]
        return batch

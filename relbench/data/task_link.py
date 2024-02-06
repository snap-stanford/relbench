import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union


import pandas as pd
from numpy.typing import NDArray
import numpy as np

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.data.task_base import BaseTask, TaskType, RelBenchBaseTask
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class LinkTask(BaseTask):
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        target_col: str,
        source_entity_table: str,
        source_entity_col: str,
        destination_entity_table: str,
        destination_entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        super().__init__(
            dataset=dataset,
            timedelta=timedelta,
            metrics=metrics,
        )

        self.target_col = target_col
        self.source_entity_table = source_entity_table
        self.source_entity_col = source_entity_col
        self.destination_entity_table = destination_entity_table
        self.destination_entity_col = destination_entity_col

        self._full_test_table = None
        self._cached_table_dict = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def filter_dangling_entities(self, table: Table) -> Table:
        src_num_entities = len(self.dataset.db.table_dict[self.source_entity_table])
        dst_num_entities = len(self.dataset.db.table_dict[self.destination_entity_table])
        
        # remove all rows where source or destination entity is out of range
        filter_mask = (table.df[self.source_entity_col] >= src_num_entities) | (table.df[self.destination_entity_col] >= dst_num_entities)

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        neg_sampling_ratio: Optional[float] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self._full_test_table

        if neg_sampling_ratio is not None:
            size_target_table_with_negs = int( (neg_sampling_ratio+1) * len(target_table))
        else:
            size_target_table_with_negs = len(target_table)
        if len(pred) != size_target_table_with_negs :
            raise ValueError(
                f"Length of pred ({len(pred)}) does not match length of target table "
                f"({size_target_table_with_negs})."
            )

        target = target_table.df[self.target_col].to_numpy()

        if neg_sampling_ratio is not None:
            target = np.concatenate((target, np.zeros(int(neg_sampling_ratio*len(target)))))

        # split pred into two arrays according to whether target = 0 or 1 
        # (i.e. whether the link exists or not)

        pred_dict = {"y_pred_pos": pred[target == 1],
                    "y_pred_neg": pred[target == 0]}

        return {fn.__name__ + str(k) if k is not None else fn.__name__: 
                fn(pred_dict, k) for fn, k in metrics} # TODO (joshrob) implement metrics


    
class RelBenchLinkTask(RelBenchBaseTask, LinkTask):
    source_entity_col: str
    source_entity_table: str
    destination_entity_col: str
    destination_entity_table: str
    time_col: str
    timedelta: pd.Timedelta
    target_col: str

    def __init__(self, dataset, process: bool = False) -> None:
        RelBenchBaseTask.__init__(self, dataset, process)
        LinkTask.__init__(self,
                          dataset=dataset,
                          timedelta=self.timedelta,
                          target_col=self.target_col,
                          source_entity_table=self.source_entity_table,
                          source_entity_col=self.source_entity_col,
                          destination_entity_table=self.destination_entity_table,
                          destination_entity_col=self.destination_entity_col,
                          metrics=self.metrics)

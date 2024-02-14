import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.data.task_base import BaseTask, TaskType, _pack_tables
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class LinkTask(BaseTask):
    r"""A link prediction task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        src_entity_table: str,
        src_entity_col: str,
        dst_entity_table: str,
        dst_entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        super().__init__(
            dataset=dataset,
            timedelta=timedelta,
            metrics=metrics,
        )

        self.src_entity_table = src_entity_table
        self.src_entity_col = src_entity_col
        self.dst_entity_table = dst_entity_table
        self.dst_entity_col = dst_entity_col

        self._full_test_table = None
        self._cached_table_dict = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def filter_dangling_entities(self, table: Table) -> Table:
        src_num_entities = len(self.dataset.db.table_dict[self.src_entity_table])
        dst_num_entities = len(self.dataset.db.table_dict[self.dst_entity_table])

        # filter dangling destination entities from a list
        table.df[self.dst_entity_col] = table.df[self.dst_entity_col].apply(
            lambda x: [i for i in x if i < dst_num_entities]
        )

        # filter dangling source entities and empty list (after above filtering)
        filter_mask = (table.df[self.src_entity_col] >= src_num_entities) | (
            ~table.df[self.dst_entity_col].map(bool)
        )

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
        raise NotImplementedError


class RelBenchLinkTask(LinkTask):
    name: str
    src_entity_col: str
    src_entity_table: str
    dst_entity_col: str
    dst_entity_table: str
    time_col: str
    timedelta: pd.Timedelta
    task_dir: str = "tasks"

    def __init__(self, dataset: str, process: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            src_entity_table=self.src_entity_table,
            src_entity_col=self.src_entity_col,
            dst_entity_table=self.dst_entity_table,
            dst_entity_col=self.dst_entity_col,
            metrics=self.metrics,
        )

        if not process:
            self.set_cached_table_dict(self.name, self.task_dir, self.dataset.name)

        def pack_tables(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
            return _pack_tables(self, root)

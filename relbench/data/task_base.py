import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class BaseTask:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        self.dataset = dataset
        self.timedelta = timedelta
        self.metrics = metrics

        self._full_test_table = None
        self._cached_table_dict = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def make_table(
        self,
        db: Database,
        timestamps: "pd.Series[pd.Timestamp]",
    ) -> Table:
        r"""To be implemented by subclass."""

        # TODO: ensure that tasks follow the right-closed convention

        raise NotImplementedError

    @property
    def train_table(self) -> Table:
        """Returns the train table for a task."""
        if "train" not in self._cached_table_dict:
            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.val_timestamp - self.timedelta,
                    self.dataset.db.min_timestamp,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["train"] = table
        else:
            table = self._cached_table_dict["train"]
        return self.filter_dangling_entities(table)

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if "val" not in self._cached_table_dict:
            table = self.make_table(
                self.dataset.db,
                pd.Series([self.dataset.val_timestamp]),
            )
            self._cached_table_dict["val"] = table
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

    def _mask_input_cols(self, table: Table) -> Table:
        input_cols = [
            table.time_col,
            *table.fkey_col_to_pkey_table.keys(),
        ]
        return Table(
            df=table.df[input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if "full_test" not in self._cached_table_dict:
            full_table = self.make_table(
                self.dataset._full_db,
                pd.Series([self.dataset.test_timestamp]),
            )
            self._cached_table_dict["full_test"] = full_table
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)
        return self._mask_input_cols(self._full_test_table)

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table."""
        raise NotImplementedError
    

    def evaluate(
        self):
        r"""Evaluate a prediction table."""
        raise NotImplementedError


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
    """
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    LINK_PREDICTION = "link_prediction"


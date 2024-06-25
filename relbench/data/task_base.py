from functools import cached_property
import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class Task:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        metrics: List[Callable[[NDArray, NDArray], float]],
        cache_dir: str | os.PathLike | None = None,
    ):
        self.dataset = dataset
        self.timedelta = timedelta
        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {timedelta}, time "
                f"diff: {time_diff})."
            )

        self.metrics = metrics
        self.cache_dir = cache_dir

        self._full_test_table = None

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

    def _make_default_train_table(self) -> Table:
        timestamps = pd.date_range(
            start=self.dataset.val_timestamp - self.timedelta,
            end=self.dataset.db.min_timestamp,
            freq=-self.timedelta,
        )
        if len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )
        table = self.make_table(self.dataset.db, timestamps)
        table = self.filter_dangling_entities(table)
        return table

    @cached_property
    def default_train_table(self) -> Table:
        """Returns the train table for a task."""
        if self.cache_dir is None:
            table = self._make_default_train_table()
        else:
            train_path = f"{self.cache_dir}/default_train"
            if not Path(train_path).exists():
                table = self._make_default_train_table()
                table.save(train_path)
            else:
                table = Table.load(train_path)
        return table

    def _make_default_val_table(self) -> Table:
        if self.dataset.val_timestamp + self.timedelta > self.dataset.db.max_timestamp:
            raise RuntimeError(
                "val timestamp + timedelta is larger than max timestamp! "
                "This would cause val labels to be generated with "
                "insufficient aggregation time."
            )

        # must stop by test_timestamp - timedelta to avoid time leakage
        end_timestamp = min(
            self.dataset.val_timestamp,
            self.dataset.test_timestamp - self.timedelta,
        )

        table = self.make_table(
            self.dataset.db,
            pd.date_range(
                self.dataset.val_timestamp,
                end_timestamp,
                freq=self.timedelta,
            ),
        )
        table = self.filter_dangling_entities(table)
        return table

    @cached_property
    def default_val_table(self) -> Table:
        if self.cache_dir is None:
            table = self._make_default_val_table()
        else:
            val_path = f"{self.cache_dir}/default_val"
            if not Path(val_path).exists():
                table = self._make_default_val_table()
                table.save(val_path)
            else:
                table = Table.load(val_path)
        return table

    def _make_test_table(self) -> Table:
        if (
            self.dataset.test_timestamp + self.timedelta
            > self.dataset._full_db.max_timestamp
        ):
            raise RuntimeError(
                "test timestamp + timedelta is larger than max timestamp! "
                "This would cause test labels to be generated with "
                "insufficient aggregation time."
            )

        # must stop by max_timestamp - timedelta
        end_timestamp = min(
            self.dataset.test_timestamp,
            self.dataset._full_db.max_timestamp - self.timedelta,
        )

        table = self.make_table(
            self.dataset._full_db,
            pd.date_range(
                self.dataset.test_timestamp,
                end_timestamp,
                freq=self.timedelta,
            ),
        )
        return table

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

    @cached_property
    def test_table(self) -> Table:
        if self.cache_dir is None:
            table = self._make_test_table()
        else:
            test_path = f"{self.cache_dir}/test"
            if not Path(test_path).exists():
                table = self._make_test_table()
                table.save(test_path)
            else:
                table = Table.load(test_path)
        self._full_test_table = table
        table = self._mask_input_cols(table)
        return table

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table."""
        raise NotImplementedError

    def evaluate(self):
        r"""Evaluate a prediction table."""
        raise NotImplementedError


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTILABEL_CLASSIFICATION: Multi-label classification task.
        LINK_PREDICTION: Link prediction task."
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LINK_PREDICTION = "link_prediction"

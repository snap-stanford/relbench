from functools import cache
import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench.data.database import Database
from relbench.data.table import Table

if TYPE_CHECKING:
    from relbench.data import Dataset


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

    def _make_table(self, split) -> Table:
        if split == "train":
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

        elif split == "val":
            if (
                self.dataset.val_timestamp + self.timedelta
                > self.dataset.db.max_timestamp
            ):
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            # must stop by test_timestamp - timedelta to avoid time leakage
            timestamps = pd.date_range(
                start=self.dataset.val_timestamp,
                end=min(
                    self.dataset.val_timestamp,
                    self.dataset.test_timestamp - self.timedelta,
                ),
                freq=self.timedelta,
            )

        elif split == "test":
            if (
                self.dataset.test_timestamp + self.timedelta
                > self.dataset._full_db.max_timestamp
            ):
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            timestamps = pd.date_range(
                start=self.dataset.test_timestamp,
                # must stop by max_timestamp - timedelta
                end=min(
                    self.dataset.test_timestamp,
                    self.dataset._full_db.max_timestamp - self.timedelta,
                ),
                freq=self.timedelta,
            )

        table = self.make_table(self.dataset.db, timestamps)
        table = self.filter_dangling_entities(table)
        return table

    def get_table(self, split) -> Table:
        """Returns the train table for a task."""
        if self.cache_dir is None:
            table = self._make_table(split)
        else:
            path = f"{self.cache_dir}/{split}"
            if not Path(path).exists():
                table = self._make_table(split)
                table.save(path)
            else:
                table = Table.load(path)
        return table

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table."""
        raise NotImplementedError

    def evaluate(self, target_table):
        r"""Evaluate a prediction table."""
        raise NotImplementedError

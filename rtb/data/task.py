import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import pandas as pd
from typing_extensions import Self

from rtb.data import Dataset


class TaskType(Enum):
    r"""The type of a task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class Task:
    r"""A task on a dataset."""

    input_cols: List[str]
    target_col: str
    task_type: TaskType
    benchmark_window_sizes: List[pd.Timedelta]
    metrics: List[str]

    def __init__(self, dataset: Dataset, window_size: pd.Timedelta):
        self.dataset = dataset
        self.window_size = window_size

    def make_table(db: Database, timestamps: pd.Series[pd.Timestamp]) -> Table:
        r"""To be implemented by subclass.

        The window is (timestamp, timestamp + window_size], i.e., left-exclusive.
        """

        # TODO: ensure that tasks follow the right-closed convention

        raise NotImplementedError

    def train_table(
        self,
        timestamps: Optional[pd.Series[pd.Timestamp]] = None,
    ) -> Table:
        """Returns the train table for a task."""

        if timestamps is None:
            # default sampler
            # traverse backwards to use the latest timestamps
            timestamps = pd.date_range(
                self.dataset.val_timestamp - self.window_size,
                self.dataset.min_timestamp,
                freq=-self.window_size,
            )

        assert timestamps.min() >= self.dataset.min_timestamp
        assert timestamps.max() + self.window_size <= self.dataset.val_timestamp

        return self.make_table(self.dataset.input_db, time_stamps)

    def val_table(
        self,
        timestamps: Optional[pd.Series[pd.Timestamp]] = None,
    ) -> Table:
        r"""Returns the val table for a task."""

        if timestamps is None:
            # default sampler
            timestamps = pd.Series([self.dataset.val_timestamp])

        assert timestamps.min() >= self.dataset.val_timestamp
        assert timestamps.max() + self.window_size <= self.dataset.test_timestamp

        return self.make_table(self.dataset.input_db, timestamps)

    def test_table(
        self,
    ) -> Table:
        r"""Returns the test table for a task."""
        timestamps = pd.Series([self.dataset.test_timestamp])

        assert timestamps.min() >= self.dataset.test_timestamp
        assert timestamps.max() + self.window_size <= self.dataset.max_timestamp

        table = self.make_table(self.dataset._full_db, timestamps)

        # only expose input columns to prevent info leakage
        table.df = table.df[task.input_cols]

        return table

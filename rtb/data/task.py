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

    min_time_col: str = "min_time"
    max_time_col: str = "max_time"

    input_cols: List[str]
    target_col: str
    task_type: TaskType
    window_sizes: List[pd.Timedelta]
    metrics: List[str]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def make_table(
        db: Database, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> Table:
        r"""To be implemented by subclass."""

        raise NotImplementedError

    def train_table(
        self,
        window_size: Optional[pd.Timedelta] = None,
        windows: Optional[
            Tuple[pd.Series[pd.Timestamp], pd.Series[pd.Timestamp]]
        ] = None,
    ) -> Table:
        """Returns the train table for a task."""

        assert window_size is None or windows is None

        if windows is None:
            if window_size is None:
                window_size = self.window_sizes[0]

            # default sampler
            windows = rolling_window_sampler(
                self.dataset.db_min_time,
                self.dataset.train_max_time,
                window_size,
                stride=window_size,
            )

        # TODO: check inclusivity-exclusivity of windows
        assert min(windows[0]) >= self.dataset.db_min_time
        assert max(windows[1]) <= self.dataset.train_max_time

        return self.make_table(self.dataset._db, windows)

    def val_table(
        self,
        window_size: Optional[pd.Timedelta] = None,
        windows: Optional[
            Tuple[pd.Series[pd.Timestamp], pd.Series[pd.Timestamp]]
        ] = None,
    ) -> Table:
        r"""Returns the val table for a task."""

        assert window_size is None or windows is None

        if windows is None:
            if window_size is None:
                window_size = self.window_sizes[0]

            # default sampler
            windows = one_window_sampler(
                self.dataset.train_max_time,
                window_size,
            )

        assert min(windows[0]) >= self.dataset.train_max_time
        assert max(windows[1]) <= self.dataset.val_max_time

        return self.make_table(self.dataset._db, windows)

    def test_table(
        self,
        window_size: Optional[pd.Timedelta] = None,
    ) -> Table:
        r"""Returns the test table for a task."""
        if window_size is None:
            window_size = self.window_sizes[0]

        windows = one_window_sampler(
            self.dataset.val_max_time,
            window_size,
        )
        table = self.make_table(self.dataset._db, windows)

        assert min(windows[0]) >= self.dataset.val_max_time
        assert max(windows[1]) <= self.dataset.db_max_time

        # only expose input columns to prevent info leakage
        table.df = table.df[task.input_cols]

        return table

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
    benchmark_timedeltas: List[pd.Timedelta]
    metrics: List[str]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @classmethod
    def make_table(
        cls,
        db: Database,
        time_df: pd.DataFrame,
    ) -> Table:
        r"""To be implemented by subclass.

        The window is (timestamp, timestamp + timedelta], i.e., left-exclusive.
        """

        # TODO: ensure that tasks follow the right-closed convention

        raise NotImplementedError

    def train_table(
        self,
        timedelta: Optional[pd.Timedelta] = None,
        time_df: Optional[pd.DataFrame] = None,
    ) -> Table:
        """Returns the train table for a task."""

        assert timedelta is None or time_df is None

        if time_df is None:
            if timedelta is None:
                # default timedelta
                timedelta = self.benchmark_timedeltas[0]

            # default sampler
            time_df = pd.DataFrame(
                dict(
                    timestamp=pd.date_range(
                        self.dataset.val_timestamp - timedelta,
                        self.dataset.min_timestamp,
                        freq=-timedelta,
                    ),
                    timedelta=timedelta,
                )
            )

        assert time_df.timestamp.min() >= self.dataset.min_timestamp
        assert (
            time_df.timestamp + time_df.timedelta
        ).max() <= self.dataset.val_timestamp

        return self.make_table(self.dataset.input_db, time_df)

    def val_table(
        self,
        timedelta: Optional[pd.Timedelta] = None,
        time_df: Optional[pd.DataFrame] = None,
    ) -> Table:
        r"""Returns the val table for a task."""

        assert timedelta is None or time_df is None

        if time_df is None:
            if timedelta is None:
                # default timedelta
                timedelta = self.benchmark_timedeltas[0]

            # default sampler
            time_df = pd.DataFrame(
                dict(timestamp=[self.dataset.val_timestamp], timedelta=timedelta),
            )

        assert time_df.timestamp.min() >= self.dataset.val_timestamp
        assert (
            time_df.timestamp + time_df.timedelta
        ).max() <= self.dataset.test_timestamp

        return self.make_table(self.dataset.input_db, time_df)

    def test_table(
        self,
        timedelta: Optional[pd.Timedelta] = None,
    ) -> Table:
        r"""Returns the test table for a task."""

        if timedelta is None:
            # default timedelta
            timedelta = self.benchmark_timedeltas[0]

        time_df = pd.DataFrame(
            dict(timestamp=[self.dataset.test_timestamp], timedelta=timedelta),
        )

        assert time_df.timestamp.min() >= self.dataset.test_timestamp
        assert (
            time_df.timestamp + time_df.timedelta
        ).max() <= self.dataset.max_timestamp

        table = self.make_table(self.dataset._full_db, time_df)

        # only expose input columns to prevent info leakage
        table.df = table.df[task.input_cols]

        return table

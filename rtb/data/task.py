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

    def __init__(self, dataset: Dataset, timedelta: pd.Timedelta):
        self.dataset = dataset
        self.timedelta = timedelta

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

    def make_test_table(self) -> Table:
        r"""Returns the test table for a task."""

        time_df = pd.DataFrame(
            dict(timestamp=[self.dataset.test_timestamp], timedelta=self.timedelta),
        )

        table = self.__class__.make_table(self.dataset._full_db, time_df)

        # only expose input columns to prevent info leakage
        table.df = table.df[task.input_cols]

        return table

    def make_default_train_table(self) -> Table:
        """Returns the train table for a task."""

        time_df = pd.DataFrame(
            dict(
                timestamp=pd.date_range(
                    self.dataset.val_timestamp - self.timedelta,
                    self.dataset.min_timestamp,
                    freq=-self.timedelta,
                ),
                timedelta=self.timedelta,
            )
        )

        return self.__class__.make_table(self.dataset.input_db, time_df)

    def make_default_val_table(self) -> Table:
        r"""Returns the val table for a task."""

        time_df = pd.DataFrame(
            dict(timestamp=[self.dataset.val_timestamp], timedelta=self.timedelta),
        )

        return self.__class__.make_table(self.dataset.input_db, time_df)

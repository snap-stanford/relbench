import os
from enum import Enum
from typing import List, Union

import pandas as pd
from typing_extensions import Self

from rtb.data.database import Database
from rtb.data.table import Table


class TaskType(Enum):
    r"""The type of a task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class Task:
    r"""A task on a database."""

    def __init__(
        self,
        input_cols: List[str],
        target_col: str,
        task_type: TaskType,
        window_sizes: List[int],
        metrics: List[str],
    ) -> None:
        r"""

        input_cols and target_col are explicit because the task table may
        contain extra columns with metadata for analysis or other purposes. To
        avoid possibility of info leakage, in the test table only the
        input_cols will be returned.

        Also, having input_cols explicit makes it easier to understand the task
        (manually or programmatically) for the user.

        Args:
            input_cols: columns to use as input
            target_col: column to use as target
            task_type: type of task
            window_sizes: window sizes used for this task in our benchmark
            metrics: metrics used for this task in our benchmark
        """

        # columns to use as input
        # only these columns will be kept in the test table
        self.input_cols = input_cols
        self.target_col = target_col  # column to use as target
        self.task_type = task_type
        self.window_sizes = window_sizes
        self.metrics = metrics

    def validate(self) -> bool:
        r"""Validate the task."""

        raise NotImplementedError

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Saves the task."""

        raise NotImplementedError

    @staticmethod
    def load(self, path: Union[str, os.PathLike]) -> Self:
        r"""Loads a task."""

        raise NotImplementedError

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""To be implemented by subclass.

        time_window_df should have columns window_min_time and window_max_time,
        containing timestamps."""

        raise NotImplementedError

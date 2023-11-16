from dataclasses import dataclass
from enum import Enum
import os
from typing_extensions import Self
from typing import Union, List

import pandas as pd

from rtb.data.table import Table
from rtb.data.database import Database


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

    def make_table(db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""To be implemented by subclass.

        time_window_df should have columns window_min_time and window_max_time,
        containing timestamps."""

        raise NotImplementedError

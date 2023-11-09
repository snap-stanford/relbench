from dataclasses import dataclass
from enum import Enum
import os
from typing_extensions import Self

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
        target_col: str,
        task_type: TaskType,
        test_time_window_sizes: list[int],
        metrics: list[str],
    ) -> None:
        self.target_col = target_col
        self.task_type = task_type
        self.test_time_window_sizes = test_time_window_sizes
        self.metrics = metrics

    def validate(self) -> bool:
        r"""Validate the task."""

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the task."""

        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Self:
        r"""Loads a task."""

        raise NotImplementedError

    def make_table(db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""To be implemented by subclass."""

        raise NotImplementedError

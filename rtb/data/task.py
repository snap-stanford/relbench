from dataclasses import dataclass
from enum import Enum
import os

import rtb


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
        metrics: list[str],
    ) -> None:
        self.label_col = label_col
        self.task_type = task_type
        self.metrics = metrics

    def validate(self) -> bool:
        r"""Validate the task."""

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the task."""

        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Task:
        r"""Loads a task."""

        raise NotImplementedError

    def create(db: rtb.data.Database, time_window_df: pd.DataFrame) -> rtb.data.Table:
        r"""To be implemented by subclass."""

        raise NotImplementedError

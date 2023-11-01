from dataclasses import dataclass
from enum import Enum
import os

import rtb


class TaskType(Enum):
    r"""The type of a task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class Task:
    r"""A task on a database."""

    table: rtb.data.table.Table
    label_col: str
    task_type: TaskType
    metrics: list[str]
    num_classes: int = 0  # number of classes for classification tasks

    def validate(self) -> bool:
        r"""Validate the task.

        Check:
        1. Lengths of time_stamps, entities[<name>], and labels match.
        2. num_classes != 0 for classification tasks.
        """

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the task."""

        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Task:
        r"""Loads a task."""

        raise NotImplementedError

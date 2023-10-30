from dataclasses import dataclass
from enum import Enum
import os


class TaskType(Enum):
    r"""The type of a task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class Task:
    r"""A task on a database. Can represent tasks on single (v1) or multiple
    (v2) entities."""

    task_type: TaskType
    metric: str
    time_stamps: list[int]  # time stamps to evaluate on
    entities: dict[str, list[str]]  # table name -> list of primary key values
    labels: list[int | float]  # list of labels
    num_classes: int = 0  # number of classes for classification tasks

    def validate(self) -> bool:
        r"""Validate the task.

        Check:
        1. Lengths of time_stamps, entities[<name>], and labels match.
        2. num_classes != 0 for classification tasks.
        """

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the task as a json file."""

        assert str(path).endswith(".json")
        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Task:
        r"""Loads a task from a json file."""

        assert str(path).endswith(".json")
        raise NotImplementedError

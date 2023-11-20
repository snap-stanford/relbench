import os
from dataclasses import dataclass
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
    r"""A task on a dataset."""

    input_cols: List[str]
    target_col: str
    task_type: TaskType
    window_sizes: List[int]
    metrics: List[str]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def make_table(db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""To be implemented by subclass.

        time_window_df should have columns window_min_time and window_max_time,
        containing timestamps."""

        raise NotImplementedError

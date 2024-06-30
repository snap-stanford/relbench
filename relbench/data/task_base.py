import hashlib
import os
import shutil
import tempfile
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench.data.database import Database
from relbench.data.dataset import Dataset
from relbench.data.table import Table


class BaseTask:
    r"""A task on a dataset."""

    timedelta: pd.Timedelta
    metrics: List[Callable[[NDArray, NDArray], float]]

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.cache_dir = cache_dir

        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {timedelta}, time "
                f"diff: {time_diff})."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def make_table(
        self,
        db: Database,
        timestamps: "pd.Series[pd.Timestamp]",
    ) -> Table:
        r"""To be implemented by subclass."""

        # TODO: ensure that tasks follow the right-closed convention

        raise NotImplementedError

    def _get_table(self, split: str) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=split != "test")

        if split == "train":
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
            freq = -self.timedelta

        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.val_timestamp
            end = min(
                self.dataset.val_timestamp
                + self.timedelta * (self.dataset.max_eval_time_frames - 1),
                self.dataset.test_timestamp - self.timedelta,
            )
            freq = self.timedelta

        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp
            end = min(
                self.dataset.test_timestamp
                + self.timedelta * (self.dataset.max_eval_time_frames - 1),
                db.max_timestamp - self.timedelta,
            )
            freq = self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)

        return table

    @lru_cache(maxsize=None)
    def get_table(self, split, mask_input_cols=None):
        if mask_input_cols is None:
            mask_input_cols = split == "test"

        table_path = f"{self.cache_dir}/{split}.parquet"
        if self.cache_dir and Path(table_path).exists():
            table = Table.load(table_path)
        else:
            table = self._get_table(split)
            if self.cache_dir:
                table.save(table_path)

        if mask_input_cols:
            table = self._mask_input_cols(table)

        return table

    def _mask_input_cols(self, table: Table) -> Table:
        input_cols = [
            table.time_col,
            *table.fkey_col_to_pkey_table.keys(),
        ]
        return Table(
            df=table.df[input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table."""
        raise NotImplementedError

    def evaluate(self):
        r"""Evaluate a prediction table."""
        raise NotImplementedError


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTILABEL_CLASSIFICATION: Multi-label classification task.
        LINK_PREDICTION: Link prediction task."
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LINK_PREDICTION = "link_prediction"


# TODO: move somewhere else
def _pack_tables(task, root: Union[str, os.PathLike]) -> Tuple[str, str]:
    _dummy_db = Database(
        table_dict={
            "train": task.train_table,
            "val": task.val_table,
            "test": task.test_table,
            "full_test": task._full_test_table,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_path = Path(tmpdir) / task.name
        _dummy_db.save(task_path)

        zip_base_path = Path(root) / task.dataset.name / task.task_dir / task.name
        zip_path = shutil.make_archive(zip_base_path, "zip", task_path)

    with open(zip_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"upload: {zip_path}")
    print(f"sha256: {sha256}")

    return f"{task.dataset.name}/{task.task_dir}/{task.name}.zip", sha256

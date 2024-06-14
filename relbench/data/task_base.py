import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class BaseTask:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        self.dataset = dataset
        self.timedelta = timedelta
        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {timedelta}, time "
                f"diff: {time_diff})."
            )

        self.metrics = metrics

        self._full_test_table = None
        self._cached_table_dict = {}

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

    @property
    def train_table(self) -> Table:
        """Returns the train table for a task."""
        if "train" not in self._cached_table_dict:
            timestamps = pd.date_range(
                start=self.dataset.val_timestamp - self.timedelta,
                end=self.dataset.train_start_timestamp or self.dataset.db.min_timestamp,
                freq=-self.timedelta,
            )
            if len(timestamps) < 3:
                raise RuntimeError(
                    f"The number of training time frames is too few. "
                    f"({len(timestamps)} given)"
                )
            table = self.make_table(
                self.dataset.db,
                timestamps,
            )
            self._cached_table_dict["train"] = table
        else:
            table = self._cached_table_dict["train"]
        return self.filter_dangling_entities(table)

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if "val" not in self._cached_table_dict:
            if (
                self.dataset.val_timestamp + self.timedelta
                > self.dataset.db.max_timestamp
            ):
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            # must stop by test_timestamp - timedelta to avoid time leakage
            end_timestamp = min(
                self.dataset.val_timestamp
                + self.timedelta * (self.dataset.max_eval_time_frames - 1),
                self.dataset.test_timestamp - self.timedelta,
            )

            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.val_timestamp,
                    end_timestamp,
                    freq=self.timedelta,
                ),
            )
            self._cached_table_dict["val"] = table
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if "full_test" not in self._cached_table_dict:
            if (
                self.dataset.test_timestamp + self.timedelta
                > self.dataset._full_db.max_timestamp
            ):
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            # must stop by max_timestamp - timedelta
            end_timestamp = min(
                self.dataset.test_timestamp
                + self.timedelta * (self.dataset.max_eval_time_frames - 1),
                self.dataset._full_db.max_timestamp - self.timedelta,
            )

            full_table = self.make_table(
                self.dataset._full_db,
                pd.date_range(
                    self.dataset.test_timestamp,
                    end_timestamp,
                    freq=self.timedelta,
                ),
            )
            self._cached_table_dict["full_test"] = full_table
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)

        return self._mask_input_cols(self._full_test_table)

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

    def set_cached_table_dict(self, task_name: str, task_dir: str, dataset_name: str):
        task_path = _pooch.fetch(
            f"{dataset_name}/{task_dir}/{task_name}.zip",
            processor=unzip_processor,
            progressbar=True,
        )

        self._cached_table_dict = Database.load(task_path).table_dict


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

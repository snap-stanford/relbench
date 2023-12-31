import hashlib
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import Dataset


class Task:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        target_col: str,
        entity_table: str,
        entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        self.dataset = dataset
        self.timedelta = timedelta
        self.target_col = target_col
        self.metrics = metrics
        self.entity_table = entity_table
        self.entity_col = entity_col

        self._full_test_table = None
        self._cached_table_dict = None

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
        if self._cached_table_dict is None:
            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.val_timestamp - self.timedelta,
                    self.dataset.db.min_timestamp,
                    freq=-self.timedelta,
                ),
            )
        else:
            table = self._cached_table_dict["train"]
        return self.filter_dangling_entities(table)

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if self._cached_table_dict is None:
            table = self.make_table(
                self.dataset.db,
                pd.Series([self.dataset.val_timestamp]),
            )
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

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

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if self._cached_table_dict is None:
            full_table = self.make_table(
                self.dataset._full_db,
                pd.Series([self.dataset.test_timestamp]),
            )
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)
        return self._mask_input_cols(self._full_test_table)

    def filter_dangling_entities(self, table: Table) -> Table:
        num_entities = len(self.dataset.db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities
        if filter_mask.any():
            table.df = table.df[~filter_mask]
        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self._full_test_table

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        return {fn.__name__: fn(target, pred) for fn in metrics}


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
    """
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


class RelBenchTask(Task):
    name: str
    task_type: TaskType
    entity_col: str
    entity_table: str
    time_col: str

    timedelta: pd.Timedelta
    target_col: str
    metrics: List[Callable[[NDArray, NDArray], float]]

    task_dir: str = "tasks"

    def __init__(self, dataset, process: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            target_col=self.target_col,
            entity_table=self.entity_table,
            entity_col=self.entity_col,
            metrics=self.metrics,
        )
        # Set cached_table_dict
        if not process:
            task_path = _pooch.fetch(
                f"{dataset.name}/{self.task_dir}/{self.name}.zip",
                processor=unzip_processor,
                progressbar=True,
            )
            self._cached_table_dict = Database.load(task_path).table_dict

    def pack_tables(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
        _dummy_db = Database(
            table_dict={
                "train": self.train_table,
                "val": self.val_table,
                "test": self.test_table,
                "full_test": self._full_test_table,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / self.name
            _dummy_db.save(task_path)

            zip_base_path = Path(root) / self.dataset.name / self.task_dir / self.name
            zip_path = shutil.make_archive(zip_base_path, "zip", task_path)

        with open(zip_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"upload: {zip_path}")
        print(f"sha256: {sha256}")

        return f"{self.dataset.name}/{self.task_dir}/{self.name}.zip", sha256

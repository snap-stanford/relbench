import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.table import Table
from relbench.utils import unzip_processor

if TYPE_CHECKING:
    from relbench.data import BenchmarkDataset, Dataset


class Task:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        target_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        self.dataset = dataset
        self.timedelta = timedelta
        self.target_col = target_col
        self.metrics = metrics

        self._full_test_table = None

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
    @lru_cache
    def train_table(self) -> Table:
        """Returns the train table for a task."""

        return self.make_table(
            self.dataset.db,
            pd.date_range(
                self.dataset.val_timestamp - self.timedelta,
                self.dataset.db.min_timestamp,
                freq=-self.timedelta,
            ),
        )

    @property
    @lru_cache
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""

        return self.make_table(
            self.dataset.db,
            pd.Series([self.dataset.val_timestamp]),
        )

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
    @lru_cache
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""

        table = self.make_table(
            self.dataset._full_db,
            pd.Series([self.dataset.test_timestamp]),
        )
        self._full_test_table = table

        return self._mask_input_cols(table)

    def evaluate(
        self,
        pred: NDArray[np.float64],
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self._full_test_table

        target = target_table.df[self.target_col].to_numpy()

        return {fn.__name__: fn(target, pred) for fn in metrics}


class RelBenchTask(Task):
    name: str
    task_type: str
    entity_col: str
    entity_table: str
    time_col: str

    timedelta: pd.Timedelta
    target_col: str
    metrics: List[Callable[[NDArray, NDArray], float]]

    task_dir: str = "tasks"
    train_table_name: str = "train"
    val_table_name: str = "val"
    test_table_name: str = "test"
    full_test_table_name: str = "full_test"

    def __init__(self, dataset: "RelBenchDataset", process: bool = False) -> None:
        self.process = process

        if not process:
            task_path = _pooch.fetch(
                f"{dataset.name}/{self.task_dir}/{self.name}.zip",
                processor=unzip_processor,
                progressbar=True,
            )
            self._dummy_db = Database.load(task_path)

        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            target_col=self.target_col,
            metrics=self.metrics,
        )

    @property
    def train_table(self) -> Table:
        if self.process:
            return super().train_table
        return self._dummy_db.table_dict[self.train_table_name]

    @property
    def val_table(self) -> Table:
        if self.process:
            return super().val_table
        return self._dummy_db.table_dict[self.val_table_name]

    @property
    def test_table(self) -> Table:
        if self.process:
            return super().test_table
        self._full_test_table = self._dummy_db.table_dict[self.full_test_table_name]
        return self._dummy_db.table_dict[self.test_table_name]

    def pack_tables(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
        _dummy_db = Database(
            table_dict={
                self.train_table_name: self.train_table,
                self.val_table_name: self.val_table,
                self.test_table_name: self.test_table,
                self.full_test_table_name: self._full_test_table,
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

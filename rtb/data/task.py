import hashlib
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rtb.data.database import Database
from rtb.data.table import Table
from rtb.utils import download_and_extract

if TYPE_CHECKING:
    from rtb.data import BenchmarkDataset, Dataset


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

        self._test_table = None

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

    def make_train_table(self) -> Table:
        """Returns the train table for a task."""

        return self.make_table(
            self.dataset.db,
            pd.date_range(
                self.dataset.val_timestamp - self.timedelta,
                self.dataset.min_timestamp,
                freq=-self.timedelta,
            ),
        )

    def make_val_table(self) -> Table:
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

    def make_test_table(self) -> Table:
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
        if target_table is None:
            target_table = self._full_test_table

        if metrics is None:
            metrics = self.metrics

        true = target_table.df[self.target_col].to_numpy()

        return {fn.__name__: fn(true, pred) for fn in metrics}


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
    test_table_name: str = "full_test"

    def __init__(self, dataset: "RelBenchDataset"):
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            target_col=self.target_col,
            metrics=self.metrics,
        )

    def pack_task(self, stage_path: Union[str, os.PathLike]) -> None:
        train_table = self.make_train_table()
        val_table = self.make_val_table()
        test_table = self.make_test_table()

        dummy_db = Database(
            table_dict={
                self.train_table_name: train_table,
                self.val_table_name: val_table,
                self.test_table_name: self._full_test_table,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / self.name
            dummy_db.save(task_path)

            zip_base_path = (
                Path(stage_path) / self.dataset.name / self.task_dir / self.name
            )
            zip_path = shutil.make_archive(zip_base_path, "zip", task_path)

        with open(zip_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"upload: {zip_path}")
        print(f"sha256: {sha256}")

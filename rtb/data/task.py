import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rtb.data.database import Database
from rtb.data.table import Table

if TYPE_CHECKING:
    from rtb.data import BenchmarkDataset, Dataset


class Task:
    r"""A task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        target_col: str,
        metrics: List[Callable[NDArray, NDArray], float],
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

    def make_train_table(self, stride: Optional[pd.Timedelta] = None) -> Table:
        """Returns the train table for a task."""

        if stride is None:
            stride = self.timedelta

        return self.make_table(
            self.dataset.input_db,
            pd.date_range(
                self.dataset.val_timestamp - self.timedelta,
                self.dataset.min_timestamp,
                freq=-stride,
            ),
        )

    def make_val_table(self) -> Table:
        r"""Returns the val table for a task."""

        return self.make_table(
            self.dataset.input_db,
            pd.Series([self.dataset.val_timestamp]),
        )

    def mask_input_cols(self, table: Table) -> Table:
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

    def make_input_test_table(self) -> Table:
        r"""Returns the test table for a task."""

        table = self.make_table(
            self.dataset._db,
            pd.Series([self.dataset.test_timestamp]),
        )
        self._test_table = table

        return self.mask_input_cols(table)

    def evaluate(
        self,
        pred: NDArray[np.float64],
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if target_table is None:
            target_table = self._test_table

        if metrics is None:
            metrics = self.benchmark_metrics

        true = target_table.df[self.target_col].to_numpy()

        return {fn.__name__: fn(true, pred) for fn in metrics}


class BenchmarkTask(Task):
    name: str
    task_type: str

    timedelta: pd.Timedelta
    target_col: str
    metrics: List[Callable[[NDArray, NDArray], float]]

    url_fmt: str = "http://relbench.stanford.edu/data/{}/tasks/{}.zip"

    task_dir: str = "tasks"
    train_file: str = "train_table.parquet"
    val_file: str = "val_table.parquet"
    test_file: str = "test_table.parquet"

    def __init__(self, dataset: "BenchmarkDataset", *, download=False, process=False):
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            target_col=self.target_col,
            metrics=self.metrics,
        )

        task_path = self.dataset.root / self.dataset.name / self.task_dir / self.name

        if process:
            assert not download

            # delete to avoid corruption
            shutil.rmtree(task_path, ignore_errors=True)
            task_path.mkdir(parents=True)

            self.train_table = super().make_train_table()
            self.train_table.save(task_path / self.train_file)

            self.val_table = super().make_val_table()
            self.val_table.save(task_path / self.val_file)

            self._test_table = super().make_input_test_table()
            self._test_table.save(task_path / self.test_file)
            self.input_test_table = self.mask_input_cols(self._test_table)

        else:
            if download:
                # delete to avoid corruption
                shutil.rmtree(task_path, ignore_errors=True)
                task_path.mkdir(parents=True)

                url = self.url_fmt.format(self.dataset.name, self.name)
                download_and_extract(url, task_path)

            self.train_table = Table.load(task_path / self.train_file)
            self.val_table = Table.load(task_path / self.val_file)
            self._test_table = Table.load(task_path / self.test_file)
            self.input_test_table = self.mask_input_cols(self._test_table)

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rtb.data.database import Database
from rtb.data.table import Table

if TYPE_CHECKING:
    from rtb.data import Dataset


class Task:
    r"""A task on a dataset."""

    task_type: str
    metrics: List[Callable[[NDArray, NDArray], float]]

    timedelta: pd.Timedelta

    target_col: str
    entity_col: str
    entity_table: str
    time_col: str = "timestamp"

    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset
        self._test_table = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    @classmethod
    def make_table(
        cls,
        db: Database,
        time_df: pd.DataFrame,
    ) -> Table:
        r"""To be implemented by subclass.

        The window is (timestamp, timestamp + timedelta], i.e., left-exclusive.
        """

        # TODO: ensure that tasks follow the right-closed convention

        raise NotImplementedError

    def make_default_train_table(self) -> Table:
        """Returns the train table for a task."""

        return self.make_table(
            self.dataset.input_db,
            pd.date_range(
                self.dataset.val_timestamp - self.timedelta,
                self.dataset.min_timestamp,
                freq=-self.timedelta,
            ),
        )

    def make_default_val_table(self) -> Table:
        r"""Returns the val table for a task."""

        return self.make_table(
            self.dataset.input_db,
            pd.Series([self.dataset.val_timestamp]),
        )

    def make_input_test_table(self) -> Table:
        r"""Returns the test table for a task."""

        table = self.make_table(
            self.dataset._db,
            pd.Series([self.dataset.test_timestamp]),
        )
        self._test_table = table

        return Table(
            # only expose input columns to prevent info leakage
            df=table.df[[self.time_col, self.entity_col]],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

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

import os
from dataclasses import dataclass
from typing import Callable, List, Union

import pandas as pd

from rtb.data import Dataset


class Task:
    r"""A task on a dataset."""

    input_cols: List[str]
    target_col: str
    task_type: str
    benchmark_timedelta_list: List[pd.Timedelta]
    benchmark_metric_dict: Dict[str, Callable[[np.ndarray, np.ndarray], float]]

    def __init__(self, dataset: Dataset, timedelta: pd.Timedelta):
        self.dataset = dataset
        self.timedelta = timedelta
        self._test_table = None

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

        time_df = pd.DataFrame(
            dict(
                timestamp=pd.date_range(
                    self.dataset.val_timestamp - self.timedelta,
                    self.dataset.min_timestamp,
                    freq=-self.timedelta,
                ),
                timedelta=self.timedelta,
            )
        )

        return self.make_table(self.dataset.input_db, time_df)

    def make_default_val_table(self) -> Table:
        r"""Returns the val table for a task."""

        time_df = pd.DataFrame(
            dict(timestamp=[self.dataset.val_timestamp], timedelta=self.timedelta),
        )

        return self.make_table(self.dataset.input_db, time_df)

    def make_input_test_table(self) -> Table:
        r"""Returns the test table for a task."""

        time_df = pd.DataFrame(
            dict(timestamp=[self.dataset.test_timestamp], timedelta=self.timedelta),
        )

        table = self.make_table(self.dataset._db, time_df)
        self._test_table = table

        return Table(
            # only expose input columns to prevent info leakage
            df=table.df[task.input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def evaluate(
        self,
        pred: np.ndarray[np.float64],
        target_table: Optional[Table] = None,
        metric_dict: Optional[
            Dict[str, Callable[[np.ndarray, np.ndarray], float]]
        ] = None,
    ) -> Dict[str, float]:
        if target_table is None:
            target_table = self._test_table

        if metric_dict is None:
            metric_dict = self.benchmark_metric_dict

        true = target_table.df[self.target_col].to_numpy()

        return {name: fn(true, pred) for name, fn in metric_dict}

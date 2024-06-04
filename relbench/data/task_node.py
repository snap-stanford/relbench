from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from relbench.data.table import Table
from relbench.data.task_base import BaseTask, TaskType, _pack_tables

if TYPE_CHECKING:
    from relbench.data import Dataset


class NodeTask(BaseTask):
    r"""A link prediction task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        target_col: str,
        entity_table: str,
        entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
    ):
        super().__init__(
            dataset=dataset,
            timedelta=timedelta,
            metrics=metrics,
        )
        self.target_col = target_col
        self.entity_table = entity_table
        self.entity_col = entity_col

        self._full_test_table = None
        self._cached_table_dict = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

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


class RelBenchNodeTask(NodeTask):
    name: str
    entity_col: str
    entity_table: str
    time_col: str
    timedelta: pd.Timedelta
    target_col: str
    task_dir: str = "tasks"

    def __init__(self, dataset: str, process: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            target_col=self.target_col,
            entity_table=self.entity_table,
            entity_col=self.entity_col,
            metrics=self.metrics,
        )

        if not process:
            self.set_cached_table_dict(self.name, self.task_dir, self.dataset.name)

        def pack_tables(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
            return _pack_tables(self, root)

    def stats(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> dict[str, dict[str, Any]]:
        r"""Get train / val / test table statistics for each timestamp
        and the whole table, including number of rows and number of entities.
        Tasks with different task types have different statistics computed:

        BINARY_CLASSIFICATION: Number of positives and negatives.
        REGRESSION: Minimum, maximum, mean, median, quantile 25 and,
            quantile 75 of the target values.
        MULTILABEL_CLASSIFICATION: Mean, minimum and maximum number of
            classes per entity. Number and index of classes having minimum
            and maximum number of classes.
        """
        if split == "train":
            table = self.train_table
        elif split == "val":
            table = self.val_table
        else:
            table = self.test_table
        timestamps = table.df[self.time_col].unique()
        res = {}
        for timestamp in timestamps:
            temp_df = table.df[table.df[self.time_col] == timestamp]
            stats = {
                "num_rows": len(temp_df),
                "num_unique_entities": temp_df[self.entity_col].nunique(),
            }
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self._set_binary_stats(temp_df, stats)
            elif self.task_type == TaskType.REGRESSION:
                self._set_regression_stats(temp_df, stats)
            elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                self._set_multilabel_stats(temp_df, stats)
            else:
                raise ValueError(f"Unsupported task type {self.task_type}")
            res[str(timestamp)] = stats
        res["total"] = {
            "total_num_rows": len(table.df),
            "total_num_unique_entities": table.df[self.entity_col].nunique(),
        }
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            self._set_binary_stats(table.df, res["total"])
        elif self.task_type == TaskType.REGRESSION:
            self._set_regression_stats(table.df, res["total"])
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            self._set_multilabel_stats(table.df, res["total"])
        else:
            raise ValueError(f"Unsupported task type {self.task_type}")
        return res

    def _set_binary_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        stats["num_positives"] = (df[self.target_col] == 1).sum()
        stats["num_negatives"] = (df[self.target_col] == 0).sum()

    def _set_regression_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        stats["min_target"] = df[self.target_col].min()
        stats["max_target"] = df[self.target_col].max()
        stats["mean_target"] = df[self.target_col].mean()
        quantiles = df[self.target_col].quantile([0.25, 0.5, 0.75])
        stats["quantile_25_target"] = quantiles.iloc[0]
        stats["median_target"] = quantiles.iloc[1]
        stats["quantile_75_target"] = quantiles.iloc[2]

    def _set_multilabel_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        arr = np.array([row for row in df[self.target_col]])
        arr_row = arr.sum(1)
        stats["mean_num_classes_per_entity"] = round(arr_row.mean(), 4)
        stats["max_num_classes_per_entity"] = arr_row.max()
        stats["min_num_classes_per_entity"] = arr_row.min()
        arr_class = arr.sum(0)
        max_num_class_idx = arr_class.argmax()
        stats["max_num_class_idx"] = max_num_class_idx
        stats["max_num_class_num"] = arr_class[max_num_class_idx]
        min_num_class_idx = arr_class.argmin()
        stats["min_num_class_idx"] = min_num_class_idx
        stats["min_num_class_num"] = arr_class[min_num_class_idx]

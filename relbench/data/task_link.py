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
from relbench.data.task_base import BaseTask, _pack_tables
from relbench.external.utils import to_unix_time

if TYPE_CHECKING:
    from relbench.data import Dataset


class LinkTask(BaseTask):
    r"""A link prediction task on a dataset."""

    def __init__(
        self,
        dataset: "Dataset",
        timedelta: pd.Timedelta,
        src_entity_table: str,
        src_entity_col: str,
        dst_entity_table: str,
        dst_entity_col: str,
        metrics: List[Callable[[NDArray, NDArray], float]],
        eval_k: int,
    ):
        super().__init__(
            dataset=dataset,
            timedelta=timedelta,
            metrics=metrics,
        )

        if self.dataset.max_eval_time_frames != 1:
            raise RuntimeError(
                "Link prediction cannot be defined over tasks with multiple "
                "eval time frames."
            )

        self.src_entity_table = src_entity_table
        self.src_entity_col = src_entity_col
        self.dst_entity_table = dst_entity_table
        self.dst_entity_col = dst_entity_col
        self.eval_k = eval_k

        self._full_test_table = None
        self._cached_table_dict = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def filter_dangling_entities(self, table: Table) -> Table:
        # filter dangling destination entities from a list
        table.df[self.dst_entity_col] = table.df[self.dst_entity_col].apply(
            lambda x: [i for i in x if i < self.num_dst_nodes]
        )

        # filter dangling source entities and empty list (after above filtering)
        filter_mask = (table.df[self.src_entity_col] >= self.num_src_nodes) | (
            ~table.df[self.dst_entity_col].map(bool)
        )

        if filter_mask.any():
            table.df = table.df[~filter_mask]
            table.df = table.df.reset_index(drop=True)

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

        expected_pred_shape = (len(target_table), self.eval_k)
        if pred.shape != expected_pred_shape:
            raise ValueError(
                f"The shape of pred must be {expected_pred_shape}, but "
                f"{pred.shape} given."
            )

        pred_isin_list = []
        dst_count_list = []
        for true_dst_nodes, pred_dst_nodes in zip(
            target_table.df[self.dst_entity_col],
            pred,
        ):
            pred_isin_list.append(
                np.isin(np.array(pred_dst_nodes), np.array(true_dst_nodes))
            )
            dst_count_list.append(len(true_dst_nodes))
        pred_isin = np.stack(pred_isin_list)
        dst_count = np.array(dst_count_list)

        return {fn.__name__: fn(pred_isin, dst_count) for fn in metrics}

    @property
    def num_src_nodes(self) -> int:
        return len(self.dataset.db.table_dict[self.src_entity_table])

    @property
    def num_dst_nodes(self) -> int:
        return len(self.dataset.db.table_dict[self.dst_entity_table])

    @property
    def val_seed_time(self) -> int:
        return to_unix_time(pd.Series([self.dataset.val_timestamp]))[0]

    @property
    def test_seed_time(self) -> int:
        return to_unix_time(pd.Series([self.dataset.test_timestamp]))[0]


class RelBenchLinkTask(LinkTask):
    name: str
    src_entity_col: str
    src_entity_table: str
    dst_entity_col: str
    dst_entity_table: str
    time_col: str
    timedelta: pd.Timedelta
    task_dir: str = "tasks"
    metrics: List[Callable[[NDArray, NDArray], float]]
    eval_k: int

    def __init__(self, dataset: str, process: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            timedelta=self.timedelta,
            src_entity_table=self.src_entity_table,
            src_entity_col=self.src_entity_col,
            dst_entity_table=self.dst_entity_table,
            dst_entity_col=self.dst_entity_col,
            metrics=self.metrics,
            eval_k=self.eval_k,
        )

        if not process:
            self.set_cached_table_dict(self.name, self.task_dir, self.dataset.name)

        def pack_tables(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
            return _pack_tables(self, root)

    def stats(self) -> dict[str, dict[str, int]]:
        r"""Get train / val / test table statistics for each timestamp
        and the whole table, including number of unique source entities,
        number of unique destination entities, number of destination
        entities and number of rows.
        """
        res = {}
        for split in ["train", "val", "test"]:
            split_stats = {}
            if split == "train":
                table = self.train_table
            elif split == "val":
                table = self.val_table
            else:
                table = self.test_table
            if table is None:
                continue
            timestamps = table.df[self.time_col].unique()
            for timestamp in timestamps:
                temp_df = table.df[table.df[self.time_col] == timestamp]
                (
                    num_unique_src_entities,
                    num_unique_dst_entities,
                    num_dst_entities,
                    num_rows,
                ) = self._get_stats(temp_df)
                split_stats[str(timestamp)] = {
                    "num_unique_src_entities": num_unique_src_entities,
                    "num_unique_dst_entities": num_unique_dst_entities,
                    "num_dst_entities": num_dst_entities,
                    "num_rows": num_rows,
                }

            (
                num_unique_src_entities,
                num_unique_dst_entities,
                num_dst_entities,
                num_rows,
            ) = self._get_stats(table.df)
            split_stats["total"] = {
                "num_unique_src_entities": num_unique_src_entities,
                "num_unique_dst_entities": num_unique_dst_entities,
                "num_dst_entities": num_dst_entities,
                "num_rows": num_rows,
            }
            res[split] = split_stats
        total_df = pd.concat(
            [
                table.df
                for table in [self.train_table, self.val_table, self.test_table]
                if table is not None
            ]
        )
        num_unique_src_entities, num_unique_dst_entities, num_dst_entities, num_rows = (
            self._get_stats(total_df)
        )
        res["total"] = {
            "num_unique_src_entities": num_unique_src_entities,
            "num_unique_dst_entities": num_unique_dst_entities,
            "num_dst_entities": num_dst_entities,
            "num_rows": num_rows,
        }
        train_uniques = set(self.train_table.df[self.src_entity_col].unique())
        if self.test_table is None:
            return res
        test_uniques = set(self.test_table.df[self.src_entity_col].unique())
        ratio_train_test_entity_overlap = len(
            train_uniques.intersection(test_uniques)
        ) / len(test_uniques)
        res["total"][
            "ratio_train_test_entity_overlap"
        ] = ratio_train_test_entity_overlap
        return res

    def _get_stats(self, df: pd.DataFrame) -> list[int]:
        num_unique_src_entities = df[self.src_entity_col].nunique()
        num_unique_dst_entities = len(
            set(value for row in df[self.dst_entity_col] for value in row)
        )
        num_dst_entities = sum(len(row) for row in df[self.dst_entity_col])
        num_rows = len(df)
        return (
            num_unique_src_entities,
            num_unique_dst_entities,
            num_dst_entities,
            num_rows,
        )

from typing import Optional
from relbench.base.dataset import Dataset
from relbench.base import Database, EntityTask, Table, TaskType

import duckdb
import pandas as pd

from relbench.metrics import (
    mae,
    r2,
    rmse,
    average_precision,
    accuracy,
    f1,
    roc_auc,
)


class PredictColumnTask(EntityTask):

    timedelta = pd.Timedelta(seconds=1)

    def __init__(
        self,
        dataset: Dataset,
        task_type: TaskType,
        entity_table: str,
        entity_col: str,
        time_col: str,
        target_col: str,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(dataset, cache_dir=cache_dir)

        self.task_type = task_type
        self.entity_table = entity_table
        self.entity_col = entity_col
        if self.entity_col is None:
            self.entity_col = "primary_key"
        self.time_col = time_col
        self.target_col = target_col

        self.num_eval_timestamps = (self.dataset.test_timestamp - self.dataset.val_timestamp).total_seconds()
        if self.task_type == TaskType.REGRESSION:
            self.metrics = [r2, mae, rmse]
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.metrics = [average_precision, accuracy, f1, roc_auc]
        else:
            raise NotImplementedError(f"Task type {self.task_type} not implemented")
        
    def filter_dangling_entities(self, table: Table) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=False)
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table
    
    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split."""

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

            start = self.dataset.test_timestamp - self.timedelta
            end = self.dataset.val_timestamp
            freq = -self.timedelta

        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = db.max_timestamp
            end = self.dataset.test_timestamp
            freq = -self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)

        return table

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        entity_table = db.table_dict[self.entity_table].df
        entity_table_removed_cols = db.table_dict[self.entity_table].removed_cols

        # Calculate minimum and maximum timestamps from timestamp_df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        min_timestamp = timestamp_df["timestamp"].min()
        max_timestamp = timestamp_df["timestamp"].max()

        df = duckdb.sql(
            f"""
            SELECT
                entity_table.{self.time_col},
                entity_table.{self.entity_col},
                entity_table_removed_cols.{self.target_col}
            FROM
                entity_table
            LEFT JOIN
                entity_table_removed_cols
            ON
                entity_table.{self.entity_col} = entity_table_removed_cols.{self.entity_col}
            WHERE
                entity_table.{self.time_col} > '{min_timestamp}' AND 
                entity_table.{self.time_col} <= '{max_timestamp}'
            """
        ).df()

        # remove rows where self.target_col is nan
        df = df.dropna(subset=[self.target_col])

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

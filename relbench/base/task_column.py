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
    r"""A predict column task on a dataset.
    Predict all values in the target column.

    The task is constructed by specifying the entity table, entity column, time column, and target column.
    The target column is removed from the entity table and saved to `db.table_dict[entity_table].removed_cols`,
    which is used to construct the table for the predict column task.

    The entity table needs to have a time column by which the data is split into training and validation set.

    Args:
        dataset: The dataset object.
        task_type: The type of the task.
        entity_table: The name of the entity table.
        entity_col: The name of the entity (id) column. Can be None if the entity table has no id column.
        time_col: The name of the time column by which the data is split into training and validation set.
        target_col: The name of the target column to be predicted.
        cache_dir: The directory to cache the task tables.
    """

    timedelta = pd.Timedelta(seconds=1)
    entity_col: str

    def __init__(
        self,
        dataset: Dataset,
        task_type: TaskType,
        entity_table: str,
        target_col: str,
        cache_dir: Optional[str] = None,
    ):

        super().__init__(dataset, cache_dir=cache_dir)

        self.task_type = task_type
        self.entity_table = entity_table
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
        r"""Helper function to get a table for a split.
        
        This function overrides the `_get_table` method in `EntityTask`.
        Because we predict all values in the target column, we only look at the min and max timestamp
        for each split and take all rows in the table between them.        
        """

        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        # Set the entity column to the pkey column of the entity table since relbench needs it
        # See ".../relbench/modeling/graph.py", task.entity_col
        self.entity_col = db.table_dict[self.entity_table].pkey_col

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

        time_col = db.table_dict[self.entity_table].time_col
        entity_col = db.table_dict[self.entity_table].pkey_col

        # Calculate minimum and maximum timestamps from timestamp_df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        min_timestamp = timestamp_df["timestamp"].min()
        max_timestamp = timestamp_df["timestamp"].max()

        df = duckdb.sql(
            f"""
            SELECT
                entity_table.{time_col},
                entity_table.{entity_col},
                entity_table_removed_cols.{self.target_col}
            FROM
                entity_table
            LEFT JOIN
                entity_table_removed_cols
            ON
                entity_table.{entity_col} = entity_table_removed_cols.{entity_col}
            WHERE
                entity_table.{time_col} > '{min_timestamp}' AND 
                entity_table.{time_col} <= '{max_timestamp}'
            """
        ).df()

        # remove rows where self.target_col is nan
        df = df.dropna(subset=[self.target_col])

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=time_col,
        )

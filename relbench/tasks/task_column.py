from typing import Dict, Optional
from relbench.base.dataset import Dataset
from relbench.base import Database, EntityTask, Table, TaskType

import duckdb
import pandas as pd

from relbench.metrics import (
    mae,
    r2,
    rmse,
)


class PredictColumnTask(EntityTask):

    timedelta = pd.Timedelta(days=1)

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Optional[str] = None,
        predict_task_config: Dict[str, str] = {},
    ):
        super().__init__(dataset, cache_dir=cache_dir)

        self.task_type = predict_task_config["task_type"]
        self.entity_table = predict_task_config["src_entity_table"]
        self.entity_col = predict_task_config["src_entity_col"]
        if self.entity_col is None:
            self.entity_col = "primary_key"
        self.time_col = predict_task_config["time_col"]
        self.target_col = predict_task_config["target_col"]

        if self.task_type == TaskType.REGRESSION:
            self.metrics = [r2, mae, rmse]
        else:
            raise NotImplementedError(f"Task type {self.task_type} not implemented")

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        entity_table = db.table_dict[self.entity_table].df
        entity_table_removed_cols = db.table_dict[self.entity_table].removed_cols
        # timestamp_df = pd.DataFrame({"timestamp": timestamps})
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
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

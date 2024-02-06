import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchTask, Table
from relbench.data.task import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window


class OutcomeTask(RelBenchTask):
    r"""Predict if a user will make any votes/posts/comments in the next 1 year."""

    name = "rel-clinicaltrial-outcome"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "outcome"
    timedelta = pd.Timedelta(days=365)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
import os
from typing import Dict

import pandas as pd
import pooch

from relbench.base import EntityTask, Table

DEFAULT_DBINFER_ADAPTER_CACHE = os.path.join(
    pooch.os_cache("relbench"), "dbinfer-adapters"
)
SYNTHETIC_TIME_COL = "__dbinfer_timestamp__"
SYNTHETIC_TIMESTAMPS: Dict[str, pd.Timestamp] = {
    "train": pd.Timestamp("1970-01-01"),
    "val": pd.Timestamp("1970-01-02"),
    "test": pd.Timestamp("1970-01-03"),
}


class DBInferTaskBase(EntityTask):
    """RelBench EntityTask backed by a DBInfer task definition."""

    dbinfer_dataset_name: str | None = None
    dbinfer_task_name: str | None = None

    timedelta = pd.Timedelta(days=1)
    metrics = []
    num_eval_timestamps = 1

    def __init__(
        self,
        dataset,
        cache_dir=None,
        adapter_cache_dir: str | None = None,
    ):
        if not self.dbinfer_dataset_name or not self.dbinfer_task_name:
            raise ValueError(
                "DBInferTaskBase subclasses must define "
                "'dbinfer_dataset_name' and 'dbinfer_task_name'."
            )

        if hasattr(dataset, "dbinfer_name"):
            if dataset.dbinfer_name != self.dbinfer_dataset_name:
                raise ValueError(
                    "Dataset and task originate from different DBInfer sources: "
                    f"{dataset.dbinfer_name} vs {self.dbinfer_dataset_name}"
                )

        if adapter_cache_dir is None:
            adapter_cache_dir = DEFAULT_DBINFER_ADAPTER_CACHE
        self._adapter_cache_dir = adapter_cache_dir
        from dbinfer_relbench_adapter.loader import load_dbinfer_data

        dataset_adapter, task_adapter = load_dbinfer_data(
            dataset_name=self.dbinfer_dataset_name,
            task_name=self.dbinfer_task_name,
            use_cache=True,
            cache_dir=self._adapter_cache_dir,
        )

        self._dataset_adapter = dataset_adapter
        self._task_adapter = task_adapter

        self.task_type = task_adapter.task_type
        self.entity_table = task_adapter.entity_table
        self.entity_col = task_adapter.entity_col
        self.target_col = task_adapter.target_col
        self.metrics = task_adapter.metrics
        self.num_labels = getattr(task_adapter, "num_labels", None)
        self.num_classes = self.num_labels

        metadata_time_col = getattr(
            task_adapter.dbinfer_task.metadata, "time_column", None
        )
        self.time_col = metadata_time_col or SYNTHETIC_TIME_COL

        super().__init__(dataset, cache_dir=cache_dir)

    def _get_table(self, split):
        if split not in SYNTHETIC_TIMESTAMPS:
            raise ValueError(f"Unknown split '{split}' for DBInfer task.")

        mock_table = self._task_adapter.get_table(split)
        df = mock_table.df.copy()

        time_col = self.time_col
        if time_col not in df.columns:
            self.time_col = SYNTHETIC_TIME_COL
            time_col = self.time_col
            df[time_col] = SYNTHETIC_TIMESTAMPS[split]
        elif time_col == SYNTHETIC_TIME_COL:
            df[time_col] = SYNTHETIC_TIMESTAMPS[split]
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        fkeys = {}
        if self.entity_col in df.columns and self.entity_table:
            fkeys[self.entity_col] = self.entity_table

        return Table(
            df=df,
            fkey_col_to_pkey_table=fkeys,
            pkey_col=None,
            time_col=time_col,
        )

    def evaluate(self, pred, table=None):
        metrics = self._task_adapter.evaluate(pred, table)
        return metrics


class AVSRepeaterTask(DBInferTaskBase):
    dbinfer_dataset_name = "avs"
    dbinfer_task_name = "repeater"


class MAGCiteTask(DBInferTaskBase):
    dbinfer_dataset_name = "mag"
    dbinfer_task_name = "cite"


class MAGVenueTask(DBInferTaskBase):
    dbinfer_dataset_name = "mag"
    dbinfer_task_name = "venue"


class DigineticaCTRTask(DBInferTaskBase):
    dbinfer_dataset_name = "diginetica"
    dbinfer_task_name = "ctr"


class DigineticaPurchaseTask(DBInferTaskBase):
    dbinfer_dataset_name = "diginetica"
    dbinfer_task_name = "purchase"


class RetailRocketCVRTask(DBInferTaskBase):
    dbinfer_dataset_name = "retailrocket"
    dbinfer_task_name = "cvr"


class SeznamChargeTask(DBInferTaskBase):
    dbinfer_dataset_name = "seznam"
    dbinfer_task_name = "charge"


class SeznamPrepayTask(DBInferTaskBase):
    dbinfer_dataset_name = "seznam"
    dbinfer_task_name = "prepay"


class AmazonRatingTask(DBInferTaskBase):
    dbinfer_dataset_name = "amazon"
    dbinfer_task_name = "rating"


class AmazonPurchaseTask(DBInferTaskBase):
    dbinfer_dataset_name = "amazon"
    dbinfer_task_name = "purchase"


class AmazonChurnTask(DBInferTaskBase):
    dbinfer_dataset_name = "amazon"
    dbinfer_task_name = "churn"


class StackExchangeChurnTask(DBInferTaskBase):
    dbinfer_dataset_name = "stackexchange"
    dbinfer_task_name = "churn"


class StackExchangeUpvoteTask(DBInferTaskBase):
    dbinfer_dataset_name = "stackexchange"
    dbinfer_task_name = "upvote"


class OutbrainCTRTask(DBInferTaskBase):
    dbinfer_dataset_name = "outbrain-small"
    dbinfer_task_name = "ctr"

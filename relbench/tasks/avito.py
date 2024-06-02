import duckdb
import pandas as pd

from relbench.data import Database, RelBenchLinkTask, RelBenchNodeTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class UserClicksTask(RelBenchNodeTask):
    r"""Predict the total number of ads each customer will click in the
    next 4 days
    """

    name = "user-clicks"
    task_type = TaskType.REGRESSION
    entity_table = "UserInfo"
    entity_col = "UserID"
    time_col = "timestamp"
    target_col = "num_click"
    timedelta = pd.Timedelta(days=4)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        search_info = db.table_dict["SearchInfo"].df
        search_stream = db.table_dict["SearchStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        df = duckdb.sql(
            f"""
            SELECT
                search_ads.UserID,
                t.timestamp,
                COUNT(search_ads.AdID) AS num_click,
            FROM
                timestamp_df t
            LEFT JOIN
            (
                    search_info
                INNER JOIN
                    search_stream
                ON
                    search_info.SearchID == search_stream.SearchID AND
                    search_stream.IsClick == 1.0
            ) search_ads
            ON
                search_ads.SearchDate > t.timestamp AND
                search_ads.SearchDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                search_ads.UserID
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"UserID": "entity_table"},
            pkey_col=None,
            time_col="timestamp",
        )


class UserAdClickTask(RelBenchLinkTask):
    r"""Predict the list of ads user will click in the next 4 days"""

    name = "user-ad-click"
    task_type = TaskType.LINK_PREDICTION
    src_entity_table = "UserInfo"
    src_entity_col = "UserID"

    dst_entity_table = "AdsInfo"
    dst_entity_col = "AdID"

    time_col = "timestamp"
    timedelta = pd.Timedelta(days=4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        search_info = db.table_dict["SearchInfo"].df
        search_stream = db.table_dict["SearchStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                search_ads.UserID,
                t.timestamp,
                LIST(search_ads.AdID) AS AdID,
            FROM
                timestamp_df t
            LEFT JOIN
            (
                    search_info
                INNER JOIN
                    search_stream
                ON
                    search_info.SearchID == search_stream.SearchID AND
                    search_stream.IsClick == 1.0
            ) search_ads
            ON
                search_ads.SearchDate > t.timestamp AND
                search_ads.SearchDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                search_ads.UserID
            """
        ).df()
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

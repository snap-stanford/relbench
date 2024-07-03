import duckdb
import pandas as pd

from relbench.data import Database, LinkTask, NodeTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class AdsClicksTask(NodeTask):
    r"""Assuming the ad will be clicked in the next 4 days, predict the
    Click-Through-Rate (CTR) for each ad.
    """

    name = "ads-clicks"
    task_type = TaskType.REGRESSION
    entity_table = "AdsInfo"
    entity_col = "AdID"
    time_col = "timestamp"
    target_col = "num_click"
    timedelta = pd.Timedelta(days=4)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        ads_info = db.table_dict["AdsInfo"].df
        search_stream = db.table_dict["SearchStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        df = duckdb.sql(
            f"""
            SELECT
                search_ads.AdID,
                t.timestamp,
                COALESCE(SUM(search_ads.isClick), 0) / COALESCE(COUNT(search_ads.SearchID), 1) AS num_click
            FROM
                timestamp_df t
            LEFT JOIN (
                ads_info
                    LEFT JOIN
                search_stream
                    ON
                ads_info.AdID == search_stream.AdID
            ) search_ads
            ON
                search_ads.SearchDate > t.timestamp AND
                search_ads.SearchDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                search_ads.AdID
            HAVING
                SUM(search_ads.isClick) > 0
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"AdID": "entity_table"},
            pkey_col=None,
            time_col="timestamp",
        )


class UserVisitsTask(NodeTask):
    r"""Predict whether each customer will visit more than one ad in the next
    4 days.
    """

    name = "user-visits"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_table = "UserInfo"
    entity_col = "UserID"
    time_col = "timestamp"
    target_col = "num_click"
    timedelta = pd.Timedelta(days=4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        user_info = db.table_dict["UserInfo"].df
        visits_stream = db.table_dict["VisitStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        df = duckdb.sql(
            f"""
            SELECT
                visit_ads.UserID,
                t.timestamp,
                COALESCE(COUNT(DISTINCT visit_ads.AdID), 0) > 1 AS num_click
            FROM
                timestamp_df t
            LEFT JOIN
            (
                    user_info
                LEFT JOIN
                    visits_stream
                ON
                    user_info.UserID == visits_stream.UserID
            ) visit_ads
            ON
                visit_ads.ViewDate > t.timestamp AND
                visit_ads.ViewDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                visit_ads.UserID
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"UserID": "entity_table"},
            pkey_col=None,
            time_col="timestamp",
        )


class UserClicksTask(NodeTask):
    r"""Predict whether the each customer will click on more than one ads in
    the next 4 days
    """

    name = "user-clicks"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_table = "UserInfo"
    entity_col = "UserID"
    time_col = "timestamp"
    target_col = "num_click"
    timedelta = pd.Timedelta(days=4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        user_info = db.table_dict["UserInfo"].df
        search_info = db.table_dict["SearchInfo"].df
        search_stream = db.table_dict["SearchStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        df = duckdb.sql(
            f"""
            SELECT
                search_ads.UserID,
                t.timestamp,
                COALESCE(COUNT(search_ads.AdID), 0) > 1 AS num_click
            FROM
                timestamp_df t
            LEFT JOIN
            (
                (
                    user_info
                        LEFT JOIN
                    search_info
                        ON
                    user_info.UserID == search_info.UserID
                ) user_search_info
                LEFT JOIN
                    search_stream
                ON
                    user_search_info.SearchID == search_stream.SearchID AND
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


class UserAdVisitTask(LinkTask):
    r"""Predict the distinct list of ads a user will visit in the next 4 days"""

    name = "user-ad-visit"
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
        user_info = db.table_dict["UserInfo"].df
        visits_stream = db.table_dict["VisitStream"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                visit_ads.UserID,
                t.timestamp,
                LIST(DISTINCT visit_ads.AdID) AS AdID,
            FROM
                timestamp_df t
            LEFT JOIN
            (
                    user_info
                LEFT JOIN
                    visits_stream
                ON
                    user_info.UserID == visits_stream.UserID
            ) visit_ads
            ON
                visit_ads.ViewDate > t.timestamp AND
                visit_ads.ViewDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                visit_ads.UserID
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

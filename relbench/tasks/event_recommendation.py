import duckdb
import pandas as pd

from relbench.data import Database, RelBenchLinkTask, RelBenchNodeTask, Table
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


class RecommendationTask(RelBenchLinkTask):
    r"""Predict the list of events a user will be interested in in the next
    15 days."""

    name = "rel-event-rec"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user"
    src_entity_table = "users"
    dst_entity_col = "event"
    dst_entity_table = "events"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=10)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 5

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        user_friends = db.table_dict["user_friends"].df
        friends = db.table_dict["friends"].df
        events = db.table_dict["events"].df
        event_attendees = db.table_dict["event_attendees"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                event_attendees.user_id AS user,
                LIST(DISTINCT event_attendees.event) AS event
            FROM
                timestamp_df t
            LEFT JOIN
                event_attendees
            ON
                event_attendees.start_time > t.timestamp AND
                event_attendees.start_time <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                event_attendees.user_id
            """
        ).df()
        df = df.dropna(subset=["user"])
        df["user"] = df["user"].astype(int)

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

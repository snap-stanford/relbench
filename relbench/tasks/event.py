import duckdb
import pandas as pd

from relbench.data import Database, RelBenchNodeTask, Table
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


class UserAttendanceTask(RelBenchNodeTask):
    r"""Predict the number of events a user will go to in the next seven days
    7 days."""

    name = "user-attendance"
    task_type = TaskType.REGRESSION
    entity_col = "user"
    entity_table = "users"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]
    target_col = "target"

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        user_friends = db.table_dict["user_friends"].df
        friends = db.table_dict["friends"].df
        events = db.table_dict["events"].df
        event_attendees = db.table_dict["event_attendees"].df
        event_interest = db.table_dict["event_interest"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                event_attendees.user_id AS user,
                SUM(CASE WHEN event_attendees.status IN ('yes', 'maybe') THEN 1 ELSE 0 END) AS target
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
        df = df.reset_index()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

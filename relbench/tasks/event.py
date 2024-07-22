import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, r2, rmse, roc_auc


class UserAttendanceTask(EntityTask):
    r"""Predict the number of events a user will go to in the next seven days 7 days."""

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
            f"""SELECT
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


class UserRepeatTask(EntityTask):
    r"""Predict whether a user will attend an event in the next 7 days if they have
    already attended an event in the last 14 days."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "user"
    entity_table = "users"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [accuracy, average_precision, f1, roc_auc]
    target_col = "target"

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        user_friends = db.table_dict["user_friends"].df
        friends = db.table_dict["friends"].df
        events = db.table_dict["events"].df
        event_attendees = db.table_dict["event_attendees"].df
        event_interest = db.table_dict["event_interest"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        eval_timestamp_len = len(timestamp_df)
        if len(timestamp_df) == 1:
            new_row = pd.DataFrame(
                {
                    "timestamp": [
                        timestamps[0] - self.timedelta * 2,
                        timestamps[0] - self.timedelta,
                    ]
                }
            )
            timestamp_df = pd.concat([new_row, timestamp_df], ignore_index=True)

        df = duckdb.sql(
            f"""
            WITH tb AS(
                SELECT
                    t.timestamp AS timestamp,
                    event_attendees.user_id AS user,
                    MAX(CASE WHEN event_attendees.status IN ('yes', 'maybe') THEN 1 ELSE 0 END) AS target,
                    MAX(MAX(CASE WHEN event_attendees.status IN ('yes', 'maybe') THEN 1 ELSE 0 END)) OVER (PARTITION BY event_attendees.user_id ORDER BY t.timestamp ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) as prev_target
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
            )
            SELECT
                timestamp,
                user,
                target
            FROM
                tb
            WHERE
                prev_target = 1;
            """
        ).df()

        if eval_timestamp_len == 1:
            df = df[df.timestamp == df.timestamp.max()]

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


class UserIgnoreTask(EntityTask):
    r"""Predict whether a user will ignore more than 2 event invitations in the next 7
    days."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "user"
    entity_table = "users"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [accuracy, average_precision, f1, roc_auc]
    target_col = "target"

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        user_friends = db.table_dict["user_friends"].df
        friends = db.table_dict["friends"].df
        events = db.table_dict["events"].df
        event_attendees = db.table_dict["event_attendees"].df
        event_interest = db.table_dict["event_interest"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        if len(timestamp_df) == 1:
            new_row = pd.DataFrame({"timestamp": [timestamps[0] - self.timedelta]})
            timestamp_df = pd.concat([new_row, timestamp_df], ignore_index=True)

        df = duckdb.sql(
            f"""SELECT
                    t.timestamp AS timestamp,
                    event_attendees.user_id AS user,
                    CASE
                        WHEN SUM(CASE WHEN event_attendees.status = 'invited' THEN 1 ELSE 0 END) > 2 THEN 1
                        ELSE 0
                    END AS target
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

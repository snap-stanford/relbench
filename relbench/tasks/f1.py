import duckdb
import pandas as pd

from relbench.base import Database, NodeTask, Table, TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, r2, rmse, roc_auc


class DriverPositionTask(NodeTask):
    r"""Predict the average finishing position of each driver
    all races in the next 2 months.
    """

    name = "driver-position"
    task_type = TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "position"
    timedelta = pd.Timedelta(days=60)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for rel-f1-position."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    mean(re.positionOrder) as position,
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverDNFTask(NodeTask):
    r"""Predict the if each driver will DNF (not finish) a race in the next 1 month."""

    name = "driver-dnf"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "did_not_finish"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for rel-f1-dnf."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    CASE
                        WHEN MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) = 1 THEN 0
                        ELSE 1
                    END AS did_not_finish
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverTop3Task(NodeTask):
    r"""Predict if each driver will qualify in the top-3 for
    a race within the next 1 month.
    """

    name = "driver-top3"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "qualifying"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for rel-f1-qualifying."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        qualifying = db.table_dict["qualifying"].df
        drivers = db.table_dict["drivers"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    CASE
                        WHEN MIN(qu.position) <= 3 THEN 1
                        ELSE 0
                    END AS qualifying
                FROM
                    timestamp_df t
                LEFT JOIN
                    qualifying qu
                ON
                    qu.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and qu.date > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    qu.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM qualifying
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        df["qualifying"] = df["qualifying"].astype("int64")

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

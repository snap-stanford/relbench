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


class DriverPositionTask(RelBenchNodeTask):
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


class DriverDNFTask(RelBenchNodeTask):
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


class DriverTop3Task(RelBenchNodeTask):
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


######## link prediction tasks ########


class DriverConstructorResultTask(RelBenchLinkTask):
    r"""Predict a list of constructors a driver will join in
    the next 10 years, depending on the F1 race results.
    """

    name = "driver-constructor-result"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "driverId"
    src_entity_table = "drivers"
    dst_entity_col = "constructorId"
    dst_entity_table = "constructors"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 10)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 5

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for DriverConstructorTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        constructors = db.table_dict["constructors"].df
        drivers = db.table_dict["drivers"].df
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                dri.driverId as driverId,
                LIST(DISTINCT re.constructorId) AS constructorId
            FROM
                timestamp_df t
            LEFT JOIN
                results re
            ON
                re.date > t.timestamp AND
                re.date <= t.timestamp + INTERVAL '{self.timedelta} days'
            LEFT JOIN
                constructors c
            ON
                c.constructorId = re.constructorId
            LEFT JOIN
                drivers dri
            ON
                dri.driverId = re.driverId
            GROUP BY
                t.timestamp,
                dri.driverId
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

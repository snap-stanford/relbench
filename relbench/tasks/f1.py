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


class PositionTask(RelBenchNodeTask):
    r"""Predict the average finishing position of each driver
    all races in the next 2 months.
    """

    name = "rel-f1-position"
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


class DidNotFinishTask(RelBenchNodeTask):
    r"""Predict the if each driver will DNF (not finish) a race in the next 1 month."""

    name = "rel-f1-dnf"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "did_not_finish"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]  # [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for results_position_next_race."""
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


class QualifyingTask(RelBenchNodeTask):
    r"""Predict if each driver will qualify in the top-3 for a race within the next 1 month."""

    name = "rel-f1-qualifying"
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


class DriverConstructorTask(RelBenchLinkTask):
    r"""Predict a list of existing constructors that a driver will
    play in the next three years according to the results table."""

    name = "rel-f1-driver-constructor"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "driverId"
    src_entity_table = "drivers"
    dst_entity_col = "constructorId"
    dst_entity_table = "constructors"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 3)
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
                d.driverId as driverId,
                LIST(DISTINCT r.constructorId) AS constructorId
            FROM
                timestamp_df t
            LEFT JOIN
                results r
            ON
                r.date > t.timestamp AND
                r.date <= t.timestamp + INTERVAL '{self.timedelta} days'
            LEFT JOIN
                constructors c
            ON
                r.constructorId = c.constructorId
            LEFT JOIN
                drivers d
            ON
                d.driverId = r.driverId
            GROUP BY
                t.timestamp,
                d.driverId
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


class DriverRaceTask(RelBenchLinkTask):
    r"""Predict a list of existing races for a driver according to
    standings in the next year."""

    name = "rel-f1-driver-race"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "driverId"
    src_entity_table = "drivers"
    dst_entity_col = "raceId"
    dst_entity_table = "races"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for DriverRaceTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        races = db.table_dict["races"].df
        drivers = db.table_dict["drivers"].df
        standings = db.table_dict["standings"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                d.driverId as driverId,
                LIST(DISTINCT r.raceId) AS raceId
            FROM
                timestamp_df t
            LEFT JOIN
                standings s
            ON
                s.date > t.timestamp AND
                s.date <= t.timestamp + INTERVAL '{self.timedelta} days'
            LEFT JOIN
                races r
            ON
                r.raceId = s.raceId
            LEFT JOIN
                drivers d
            ON
                d.driverId = s.driverId
            GROUP BY
                t.timestamp,
                d.driverId
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

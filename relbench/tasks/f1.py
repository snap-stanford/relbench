import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
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


class DriverPositionTask(EntityTask):
    r"""Predict the average finishing position of each driver all races in the next 2
    months."""

    task_type = TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "position"
    timedelta = pd.Timedelta(days=60)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    re.driverId as driverId,
                    mean(re.positionOrder) as position,
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                WHERE
                    re.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, re.driverId

            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverDNFTask(EntityTask):
    r"""Predict the if each driver will DNF (not finish) a race in the next 1 month."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "did_not_finish"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    re.driverId as driverId,
                    MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                WHERE
                    re.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, re.driverId

            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverTop3Task(EntityTask):
    r"""Predict if each driver will qualify in the top-3 for a race within the next 1
    month."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "qualifying"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        qualifying = db.table_dict["qualifying"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    qu.driverId as driverId,
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
                WHERE
                    qu.driverId IN (
                        SELECT DISTINCT driverId
                        FROM qualifying
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, qu.driverId

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


class DriverRaceCompeteTask(RecommendationTask):
    r"""Predict in which races a driver will compete in the next 1 year."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "driverId"
    src_entity_table = "drivers"
    dst_entity_col = "raceId"
    dst_entity_table = "races"
    target_col = "raceId"
    time_col = "date"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    re.driverId as driverId,
                    LIST(DISTINCT re.raceId) as raceId
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date > t.timestamp
                GROUP BY t.timestamp, re.driverId
            ;
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

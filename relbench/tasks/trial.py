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


class StudyOutcomeTask(EntityTask):
    r"""Predict if the trials in the next 1 year will achieve its primary outcome."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "outcome"
    timedelta = pd.Timedelta(days=365)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        studies = db.table_dict["studies"].df
        outcomes = db.table_dict["outcomes"].df
        outcome_analyses = db.table_dict["outcome_analyses"].df

        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    oa.nct_id,
                    oa.p_value,
                    s.start_date,
                    oa.date
                FROM outcome_analyses oa
                LEFT JOIN outcomes o
                ON oa.outcome_id = o.id
                LEFT JOIN studies s
                ON s.nct_id = o.nct_id
                where (oa.p_value_modifier is null or oa.p_value_modifier != '>')
                and oa.p_value >=0
                and oa.p_value <=1
                and o.outcome_type = 'Primary'
            )

            SELECT
                t.timestamp,
                tr.nct_id,
                CASE
                    WHEN MIN(tr.p_value) <= 0.05 THEN 1
                    ELSE 0
                END AS outcome
            FROM timestamp_df t
            LEFT JOIN TRIAL_INFO tr
            ON tr.start_date <= t.timestamp
                and tr.date > t.timestamp
                and tr.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE tr.nct_id is not null
            GROUP BY t.timestamp, tr.nct_id;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class StudyAdverseTask(EntityTask):
    r"""Predict the number of affected patients with severe advsere events/death for the
    trial in the next 1 year."""

    task_type = TaskType.REGRESSION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "num_of_adverse_events"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        reported_event_totals = db.table_dict["reported_event_totals"].df
        studies = db.table_dict["studies"].df

        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    r.nct_id,
                    r.event_type,
                    r.subjects_affected,
                    r.date,
                    s.start_date
                FROM reported_event_totals r
                LEFT JOIN studies s
                ON r.nct_id = s.nct_id
                WHERE r.event_type = 'serious' or r.event_type = 'deaths'
                and r.subjects_affected is not null
            )

            SELECT
                t.timestamp,
                tr.nct_id,
                sum(tr.subjects_affected) AS num_of_adverse_events
            FROM timestamp_df t
            LEFT JOIN TRIAL_INFO tr
            ON tr.start_date <= t.timestamp
                and tr.date > t.timestamp
                and tr.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE tr.nct_id is not null and tr.subjects_affected is not null
            GROUP BY t.timestamp, tr.nct_id;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class SiteSuccessTask(EntityTask):
    r"""Predict the success rate of a trial site in the next 1 year."""

    task_type = TaskType.REGRESSION
    entity_col = "facility_id"
    entity_table = "facilities"
    time_col = "timestamp"
    target_col = "success_rate"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        facilities = db.table_dict["facilities"].df
        facility_study = db.table_dict["facilities_studies"].df
        outcome_analyses = db.table_dict["outcome_analyses"].df
        studies = db.table_dict["studies"].df
        outcomes = db.table_dict["outcomes"].df

        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    oa.nct_id,
                    MIN(CASE WHEN oa.p_value < 0.05 THEN 1 ELSE 0 END) AS is_successful, -- Determine if the trial is successful
                    oa.date,
                FROM outcome_analyses oa
                LEFT JOIN outcomes o
                ON oa.outcome_id = o.id
                WHERE (oa.p_value_modifier is null or oa.p_value_modifier != '>')
                and oa.p_value >=0
                and oa.p_value <=1
                and o.outcome_type = 'Primary'
                GROUP BY oa.nct_id, oa.date
            )

            SELECT
                t.timestamp,
                fs.facility_id,
                SUM(tr.is_successful)/COUNT(tr.is_successful) AS success_rate
            FROM timestamp_df t
            LEFT JOIN TRIAL_INFO tr
            LEFT JOIN facility_study fs ON fs.nct_id = tr.nct_id
            ON tr.date > t.timestamp
                and tr.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE fs.facility_id is not null
            GROUP BY t.timestamp, fs.facility_id;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ConditionSponsorRunTask(RecommendationTask):
    r"""Predict whether this condition will have which sponsors."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "condition_id"
    src_entity_table = "conditions"
    dst_entity_col = "sponsor_id"
    dst_entity_table = "sponsors"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        sponsors_studies = db.table_dict["sponsors_studies"].df
        condition_study = db.table_dict["conditions_studies"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                cs.condition_id,
                LIST(DISTINCT ss.sponsor_id) AS sponsor_id
            FROM timestamp_df t
            LEFT JOIN condition_study cs
            LEFT JOIN sponsors_studies ss ON ss.nct_id = cs.nct_id
            ON cs.date > t.timestamp
                and cs.date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY t.timestamp, cs.condition_id;
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


class SiteSponsorRunTask(RecommendationTask):
    r"""Predict whether this sponsor will have a trial in a facility."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "facility_id"
    src_entity_table = "facilities"
    dst_entity_col = "sponsor_id"
    dst_entity_table = "sponsors"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        sponsors_studies = db.table_dict["sponsors_studies"].df
        facility_study = db.table_dict["facilities_studies"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                fs.facility_id,
                LIST(DISTINCT ss.sponsor_id) AS sponsor_id
            FROM timestamp_df t
            LEFT JOIN facility_study fs
            LEFT JOIN sponsors_studies ss ON ss.nct_id = fs.nct_id
            ON fs.date > t.timestamp
                and fs.date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY t.timestamp, fs.facility_id;
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

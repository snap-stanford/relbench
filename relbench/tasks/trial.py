import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchTask, Table
from relbench.data.task import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, multilabel_f1_micro, multilabel_f1_macro
from relbench.utils import get_df_in_window


class OutcomeTask(RelBenchTask):
    r"""Predict if a trial will achieve its primary outcome."""

    name = "rel-trial-outcome"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "outcome"
    timedelta = pd.Timedelta(days=365)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    oa.nct_id,
                    oa.p_value,
                    s.start_date,
                    oa.inferred_date
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
                and tr.inferred_date > t.timestamp
                and tr.inferred_date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY t.timestamp, tr.nct_id;
            """ 
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class AdverseEventTask(RelBenchTask):
    r"""Predict the number of affected patients with severe advsere events/death for the trial."""

    name = "rel-trial-adverse"
    task_type = TaskType.REGRESSION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "num_of_adverse_events"
    timedelta = pd.Timedelta(days=365)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    r.nct_id,
                    r.event_type,
                    r.subjects_affected,
                    r.inferred_date,
                    s.start_date
                FROM reported_event_totals r
                LEFT JOIN studies s
                ON r.nct_id = s.nct_id
                WHERE r.event_type = 'serious' or r.event_type = 'deaths'
            )
        
            SELECT
                t.timestamp,
                tr.nct_id,
                sum(tr.subjects_affected) AS num_of_adverse_events
            FROM timestamp_df t
            LEFT JOIN TRIAL_INFO tr
            ON tr.start_date <= t.timestamp
                and tr.inferred_date > t.timestamp
                and tr.inferred_date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY t.timestamp, tr.nct_id;
            """ 
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class WithdrawalTask(RelBenchTask):
    r"""Predict the the set of reasons of withdrawals for each trial"""

    name = "rel-trial-withdrawal"
    task_type = TaskType.MULTILABEL_CLASSIFICATION
    entity_col = "nct_id"
    entity_table = "studies"
    time_col = "timestamp"
    target_col = "withdraw_reasons"
    timedelta = pd.Timedelta(days=365)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        df = duckdb.sql(
            f"""
            WITH TRIAL_INFO AS (
                SELECT
                    d.nct_id,
                    d.reason,
                    s.start_date,
                    d.inferred_date
                FROM drop_withdrawals d
                LEFT JOIN studies s
                ON s.nct_id = d.nct_id 
                WHERE d.reason IN ('Withdrawal by Subject', 'Adverse Event', 'Lost to Follow-up',
               'Protocol Violation', 'Death', 'Physician Decision', 'Lack of Efficacy',
               'Pregnancy', 'Progressive Disease', 'Disease Progression',
               'Sponsor Decision', 'Disease progression', 'Progressive disease',
               'Study Terminated by Sponsor', 'Non-compliance')
            )
        
            SELECT
                t.timestamp,
                tr.nct_id,
                string_agg(tr.reason, ',') AS withdraw_reasons
            FROM timestamp_df t
            LEFT JOIN TRIAL_INFO tr
            ON tr.start_date <= t.timestamp
                and tr.inferred_date > t.timestamp
                and tr.inferred_date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY t.timestamp, tr.nct_id;
            """ 
        ).df()
        import numpy as np
        reasons = ['Withdrawal by Subject', 'Adverse Event', 'Lost to Follow-up',
        'Protocol Violation', 'Death', 'Physician Decision', 'Lack of Efficacy',
        'Pregnancy', 'Progressive Disease', 'Disease Progression',
        'Sponsor Decision', 'Disease progression', 'Progressive disease',
        'Study Terminated by Sponsor', 'Non-compliance']
        labels = range(len(reasons))
        self.label2reason = dict(zip(reasons, labels))
        
        def map_reasons(x):
            return np.unique([self.label2reason[i] for i in x.split(',')]).tolist()

        df['withdraw_reasons'] = df.withdraw_reasons.apply(lambda x: map_reasons(x))
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    def get_label_meaning(self):
        return self.label2reason
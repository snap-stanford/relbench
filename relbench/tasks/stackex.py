import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchTask, Table
from relbench.data.task import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window


class EngageTask(RelBenchTask):
    r"""Predict if a user will make any votes/posts/comments in the next 3 years."""

    name = "rel-stackex-engage"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "OwnerUserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "contribution"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserContributionTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        comments = db.table_dict["comments"].df
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df
        users = db.table_dict["users"].df

        df = duckdb.sql(
            f"""
            WITH
            ALL_ENGAGEMENT AS (
                SELECT
                    p.id,
                    p.owneruserid as userid,
                    p.creationdate
                FROM posts p
                UNION
                SELECT
                    v.id,
                    v.userid,
                    v.creationdate
                FROM votes v
                UNION
                SELECT
                    c.id,
                    c.userid,
                    c.creationdate
                FROM comments c
            ),

            ACTIVE_USERS AS (
                 SELECT
                    t.timestamp,
                    u.id,
                    count(distinct a.id) as n_engagement
                FROM timestamp_df t
                CROSS JOIN users u
                LEFT JOIN all_engagement a
                ON u.id = a.UserId
                    and a.CreationDate <= t.timestamp
                WHERE u.id != -1
                GROUP BY t.timestamp, u.id
             )
                SELECT
                    u.timestamp,
                    u.id as OwnerUserId,
                    IF(count(distinct a.id) >= 1, 1, 0) as contribution
                FROM active_users u
                LEFT JOIN all_engagement a
                ON u.id = a.UserId
                    and a.CreationDate > u.timestamp
                    and a.CreationDate <= u.timestamp + INTERVAL '{self.timedelta}'
                where u.n_engagement >= 1
                GROUP BY u.timestamp, u.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class VotesTask(RelBenchTask):
    r"""Predict the number of upvotes that a question that is posted within the
    last 2 years will receive in the next 6 months. ?"""
    name = "rel-stackex-votes"
    task_type = TaskType.REGRESSION
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=180)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for post_votes_next_month."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    p.id as PostId,
                    count(distinct v.id) as popularity
                FROM timestamp_df t
                LEFT JOIN posts p
                ON p.CreationDate > t.timestamp - INTERVAL '730 days'
                and p.CreationDate <= t.timestamp
                and p.owneruserid != -1
                and p.owneruserid is not null
                and p.PostTypeId = 1
                LEFT JOIN votes v
                ON p.id = v.PostId
                and v.CreationDate > t.timestamp
                and v.CreationDate <= t.timestamp + INTERVAL '{self.timedelta}'
                and v.votetypeid = 2
                GROUP BY t.timestamp, p.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

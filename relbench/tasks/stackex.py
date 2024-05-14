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
    rmse,
    roc_auc,
)
from relbench.utils import get_df_in_window

######## node prediction tasks ########


class EngageTask(RelBenchNodeTask):
    r"""Predict if a user will make any votes/posts/comments in the next 2 years."""

    name = "rel-stackex-engage"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "OwnerUserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "contribution"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
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
                FROM
                    posts p
                UNION
                SELECT
                    v.id,
                    v.userid,
                    v.creationdate
                FROM
                    votes v
                UNION
                SELECT
                    c.id,
                    c.userid,
                    c.creationdate
                FROM
                    comments c
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
                FROM
                    active_users u
                LEFT JOIN
                    all_engagement a
                ON
                    u.id = a.UserId AND
                    a.CreationDate > u.timestamp AND
                    a.CreationDate <= u.timestamp + INTERVAL '{self.timedelta}'
                where
                    u.n_engagement >= 1
                GROUP BY
                    u.timestamp, u.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class VotesTask(RelBenchNodeTask):
    r"""Predict the number of upvotes that an existing question will receive in
    the next 2 years."""

    name = "rel-stackex-votes"
    task_type = TaskType.REGRESSION
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                p.id AS PostId,
                COUNT(distinct v.id) AS popularity
            FROM
                timestamp_df t
            LEFT JOIN
                posts p
            ON
                p.CreationDate <= t.timestamp AND
                p.owneruserid != -1 AND
                p.owneruserid is not null AND
                p.PostTypeId = 1
            LEFT JOIN
                votes v
            ON
                p.id = v.PostId AND
                v.CreationDate > t.timestamp AND
                v.CreationDate <= t.timestamp + INTERVAL '{self.timedelta}' AND
                v.votetypeid = 2
            GROUP BY
                t.timestamp,
                p.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class BadgesTask(RelBenchNodeTask):
    r"""Predict if each user will receive in a new badge the next 2 years."""

    name = "rel-stackex-badges"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "UserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "WillGetBadge"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        users = db.table_dict["users"].df
        badges = db.table_dict["badges"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                u.Id as UserId,
            CASE WHEN
                COUNT(b.Id) >= 1 THEN 1 ELSE 0 END AS WillGetBadge
            FROM
                timestamp_df t
            LEFT JOIN
                users u
            ON
                u.CreationDate <= t.timestamp
            LEFT JOIN
                badges b
            ON
                u.Id = b.UserID
                AND b.Date > t.timestamp
                AND b.Date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY
                t.timestamp,
                u.Id
            """
        ).df()

        # remove any IderId rows that are NaN
        df = df.dropna(subset=["UserId"])
        df[self.entity_col] = df[self.entity_col].astype(
            int
        )  # for some reason duckdb returns float64 keys

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


######## link prediction tasks ########


class UserCommentOnPostTask(RelBenchLinkTask):
    r"""Predict a list of existing posts that a user will comment in the next
    two years."""

    name = "rel-stackex-comment-on-post"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "UserId"
    src_entity_table = "users"
    dst_entity_col = "PostId"
    dst_entity_table = "posts"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserCommentOnPostTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        users = db.table_dict["users"].df
        posts = db.table_dict["posts"].df
        comments = db.table_dict["comments"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                c.UserId as UserId,
                LIST(DISTINCT p.id) AS PostId
            FROM
                timestamp_df t
            LEFT JOIN
                posts p
            ON
                p.CreationDate <= t.timestamp
            LEFT JOIN
                comments c
            ON
                p.id = c.PostId AND
                c.CreationDate > t.timestamp AND
                c.CreationDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                c.UserId is not null AND
                p.owneruserid != -1 AND
                p.owneruserid is not null
            GROUP BY
                t.timestamp,
                c.UserId
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


class RelatedPostTask(RelBenchLinkTask):
    r"""Predict a list of existing posts that users will link a given post to in the next
    two years."""

    name = "rel-stackex-related-post"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "PostId"
    src_entity_table = "posts"
    dst_entity_col = "postLinksIdList"
    dst_entity_table = "posts"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserVoteOnPostTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        posts = db.table_dict["posts"].df
        postLinks = db.table_dict["postLinks"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    pl.PostId as PostId,
                    LIST(DISTINCT pl.RelatedPostId) AS postLinksIdList
                FROM
                    timestamp_df t
                LEFT JOIN
                    postLinks pl
                ON
                    pl.CreationDate > t.timestamp AND
                    pl.CreationDate <= t.timestamp + INTERVAL '{self.timedelta} days'
                LEFT JOIN
                    posts p1
                ON
                    pl.PostId = p1.Id
                LEFT JOIN
                    posts p2
                ON
                    pl.RelatedPostId = p2.Id
                WHERE
                    pl.PostId IS NOT NULL AND
                    pl.RelatedPostId IS NOT NULL AND
                    p1.CreationDate <= t.timestamp AND
                    p2.CreationDate <= t.timestamp
                GROUP BY
                    t.timestamp,
                    pl.PostId;
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


class UsersInteractTask(RelBenchLinkTask):
    r"""Predict a list of users who comment on the same posts as the original user."""

    name = "rel-stackex-users-interact"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "UserId"
    src_entity_table = "users"
    dst_entity_col = "InteractingUserId"
    dst_entity_table = "users"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)

    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UsersInteractTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        users = db.table_dict["users"].df
        posts = db.table_dict["posts"].df
        comments = db.table_dict["comments"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                c.UserId AS UserId,
                LIST(DISTINCT c2.UserId) FILTER (WHERE c2.UserId IS NOT NULL) AS InteractingUserId
            FROM
                timestamp_df t
            CROSS JOIN
                comments c
            JOIN
                users u_c ON c.UserId = u_c.Id
                AND u_c.CreationDate < t.timestamp
            LEFT JOIN
                comments c2 ON c.postid = c2.postid
            JOIN
                users u_c2 ON c2.UserId = u_c2.Id AND u_c2.CreationDate < t.timestamp
                    AND c2.UserId != c.UserId AND c2.UserId IS NOT NULL
                    AND c2.CreationDate > t.timestamp
                    AND c2.CreationDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                c.UserId IS NOT NULL
                AND NOT (c.UserId IS NULL OR c2.UserId IS NULL)
            GROUP BY
                t.timestamp,
                c.UserId
            HAVING
                ARRAY_LENGTH(ARRAY_AGG(DISTINCT c2.UserId)) > 0;
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

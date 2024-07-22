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

######## node prediction tasks ########


class UserEngagementTask(EntityTask):
    r"""Predict if a user will make any votes/posts/comments in the next 2 years."""

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


class PostVotesTask(EntityTask):
    r"""Predict the number of upvotes that an existing question will receive in the next
    2 years."""

    task_type = TaskType.REGRESSION
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]

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


class UserBadgeTask(EntityTask):
    r"""Predict if each user will receive in a new badge the next 2 years."""

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


class UserPostCommentTask(RecommendationTask):
    r"""Predict a list of existing posts that a user will comment in the next two
    years."""

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


class PostPostRelatedTask(RecommendationTask):
    r"""Predict a list of existing posts that users will link a given post to in the
    next two years."""

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

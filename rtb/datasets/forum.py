import os

from typing import Dict, Tuple
import pandas as pd

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset
from rtb.utils import to_unix_time


class user_posts_next_three_months(Task):
    r"""Predict the number of posts a user will make in the next 3 months."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "OwnerUserId"],
            target_col="num_posts",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta(days=90)],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for user_posts_next_three_months."""

        posts = db.tables["posts"].df
        posts = posts[
            posts.OwnerUserId != -1
        ]  ## when user id is -1, it is stats exchange community, not a real person
        posts = posts[posts.OwnerUserId.notnull()]  ## 1153 null posts
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                OwnerUserId,
                COUNT(Id) AS num_posts
            FROM
                time_window_df,
                (
                    SELECT
                        OwnerUserId,
                        CreaionDate,
                        Id
                    FROM
                        posts
                ) AS tmp
            WHERE
                tmp.CreaionDate > time_window_df.window_min_time AND
                tmp.CreaionDate <= time_window_df.window_max_time
            GROUP BY OwnerUserId, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"OwnerUserId": "users"},
            pkey_col=None,
            time_col="window_min_time",
        )


class comment_scores_next_six_months(Task):
    r"""Predict the sum of scores of comments that a user will make in the next 6 months."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "UserId"],
            target_col="comment_scores",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta(days=180)],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for post_next_three_months."""

        comments = db.tables["comments"].df
        comments = comments[
            comments.UserId != -1
        ]  ## when user id is -1, it is stats exchange community, not a real person
        comments = comments[comments.UserId.notnull()]  ## 2439 null comments
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                UserId,
                SUM(Score) AS comment_scores
            FROM
                time_window_df,
                (
                    SELECT
                        UserId,
                        CreationDate,
                        Score
                    FROM
                        comments
                ) AS tmp
            WHERE
                tmp.CreationDate > time_window_df.window_min_time AND
                tmp.CreationDate <= time_window_df.window_max_time
            GROUP BY UserId, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"UserId": "users"},
            pkey_col=None,
            time_col="window_min_time",
        )


class post_upvotes_next_week(Task):
    r"""Predict the number of upvotes that a post will receive in the next week."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "PostId"],
            target_col="num_upvotes",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta("1W")],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for post_votes_next_month."""

        votes = db.tables["votes"].df
        votes = votes[votes.PostId.notnull()]
        votes = votes[votes.VoteTypeId == 2]  ## upvotes
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                PostId,
                COUNT(Id) AS num_upvotes
            FROM
                time_window_df,
                (
                    SELECT
                        PostId,
                        CreationDate,
                        Id
                    FROM
                        votes
                ) AS tmp
            WHERE
                tmp.CreationDate > time_window_df.window_min_time AND
                tmp.CreationDate <= time_window_df.window_max_time
            GROUP BY PostId, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"PostId": "posts"},
            pkey_col=None,
            time_col="window_min_time",
        )


class ForumDataset(Dataset):
    name = "rtb-forum"

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""
        ## needs to brainstorm a bit about meaningful tasks

        tasks = {
            "user_posts_next_three_months": user_posts_next_three_months(),
            "comment_scores_next_six_months": comment_scores_next_six_months(),
            "post_upvotes_next_week": post_upvotes_next_week(),
        }
        self.tasks_window_size = {i: j.window_sizes[0] for i, j in tasks.items()}
        return tasks

    def process(self) -> Database:
        r"""Process the raw files into a database."""
        path = f"{self.root}/{self.name}/raw/"
        badges = pd.read_csv(os.path.join(path, "badges.csv"))
        comments = pd.read_csv(os.path.join(path, "comments.csv"))
        postHistory = pd.read_csv(os.path.join(path, "postHistory.csv"))
        postLinks = pd.read_csv(os.path.join(path, "postLinks.csv"))
        posts = pd.read_csv(os.path.join(path, "posts.csv"))
        # tags = pd.read_csv(os.path.join(path, "tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns
        users = pd.read_csv(os.path.join(path, "users.csv"))
        votes = pd.read_csv(os.path.join(path, "votes.csv"))

        ## remove time leakage columns
        users.drop(
            columns=["Views", "UpVotes", "DownVotes", "LastAccessDate"], inplace=True
        )
        posts.drop(columns=["LasActivityDate"], inplace=True)

        ## change time column to unix time
        comments["CreationDate"] = to_unix_time(comments["CreationDate"])
        badges["Date"] = to_unix_time(badges["Date"])
        postLinks["CreationDate"] = to_unix_time(postLinks["CreationDate"])

        postHistory["CreationDate"] = to_unix_time(postHistory["CreationDate"])
        votes["CreationDate"] = to_unix_time(votes["CreationDate"])
        posts["CreaionDate"] = to_unix_time(posts["CreaionDate"])

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            fkey_col_to_pkey_table={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkey_col_to_pkey_table={
                "UserId": "users",
            },
            pkey_col="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "RelatedPostId": "posts",  ## is this allowed? two foreign keys into the same primary
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col=None,
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "LastEditorUserId": "users",
                "ParentId": "posts",  # notice the self-reference
            },
            pkey_col="Id",
            time_col="CreaionDate",
        )

        return Database(tables)

    def get_cutoff_times(self) -> Tuple[int, int]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        train_max_time = pd.Timestamp(
            "2013-09-14"
        )  # 2013-09-14 02:54:13, 6+6 months before the max_time
        val_max_time = pd.Timestamp(
            "2014-03-14"
        )  # 2014-03-14 02:54:13, 6 months before the max_time
        return train_max_time, val_max_time

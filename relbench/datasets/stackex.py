import os

import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.stackex import (
    BadgesTask,
    EngageTask,
    RelatedPostTask,
    UserCommentOnPostTask,
    VotesTask,
)
from relbench.utils import unzip_and_convert_csv_to_parquet_processor


class StackExDataset(RelBenchDataset):
    name = "rel-stackex"
    # 2 years gap
    val_timestamp = pd.Timestamp("2019-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")
    max_eval_time_frames = 1
    task_cls_list = [
        EngageTask,
        VotesTask,
        BadgesTask,
        UserCommentOnPostTask,
        RelatedPostTask,
    ]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_and_convert_csv_to_parquet_processor,
        )
        path = os.path.join(path, "raw")
        users_cols = ('Id', 'AccountId', 'DisplayName', 'Location',
                      'ProfileImageUrl', 'WebsiteUrl', 'AboutMe',
                      'CreationDate')
        comments_cols = ('Id', 'PostId', 'UserId', 'ContentLicense',
                         'UserDisplayName', 'Text', 'CreationDate')
        posts_cols = ('Id', 'OwnerUserId', 'PostTypeId', 'AcceptedAnswerId',
                      'ParentId', 'OwnerDisplayName', 'Title', 'Tags',
                      'ContentLicense', 'Body', 'CreationDate')
        votes_cols = ('Id', 'UserId', 'PostId', 'VoteTypeId', 'CreationDate')
        read_parquet = lambda fname, col_names: pd.read_parquet(
            os.path.join(path, fname), columns=col_names, engine='fastparquet')

        users = read_parquet("Users.parquet", users_cols)
        comments = read_parquet("Comments.parquet", comments_cols)
        posts = read_parquet("Posts.parquet", posts_cols)
        votes = read_parquet("Votes.parquet", votes_cols)
        postLinks = read_parquet("PostLinks.parquet", None)
        badges = read_parquet("Badges.parquet", None)
        postHistory = read_parquet("PostHistory.parquet", None)

        # we remove tag table here since after removing time leakage columns,
        # all information are kept in the posts tags columns
        # tags = read_parquet("Tags.parquet", None)

        ## change time column to pd timestamp series
        comments["CreationDate"] = pd.to_datetime(comments["CreationDate"])
        badges["Date"] = pd.to_datetime(badges["Date"])
        postLinks["CreationDate"] = pd.to_datetime(postLinks["CreationDate"])

        postHistory["CreationDate"] = pd.to_datetime(postHistory["CreationDate"])
        votes["CreationDate"] = pd.to_datetime(votes["CreationDate"])
        posts["CreationDate"] = pd.to_datetime(posts["CreationDate"])
        users["CreationDate"] = pd.to_datetime(users["CreationDate"])

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
                # is this allowed? two foreign keys into the same primary
                "RelatedPostId": "posts",
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
            time_col="CreationDate",
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "ParentId": "posts",  # notice the self-reference
                "AcceptedAnswerId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        return Database(tables)

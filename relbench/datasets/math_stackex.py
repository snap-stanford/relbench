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
from relbench.utils import clean_datetime, unzip_processor


class MathStackExDataset(RelBenchDataset):
    name = "rel-math-stackex"
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
        cache_dir: str = None,
    ):
        self.cache_dir = cache_dir
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-stackex-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="31003ec800eee341bc093549b10bff8e4394fd6bc0429769a71a42b8addd1765",
            progressbar=True,
            processor=unzip_processor,
            path=self.cache_dir,
        )
        path = os.path.join(path, "math-stackex-temp")
        print("Loading data from:", path)
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(
            os.path.join(path, "Comments.csv"), low_memory=False, lineterminator="\n"
        )
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))

        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(
            os.path.join(path, "PostHistory.csv"), low_memory=False, lineterminator="\n"
        )
        print("Data loaded")

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        ## remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes", "DownVotes", "LastAccessDate"],
            inplace=True,
        )

        posts.drop(
            columns=[
                "ViewCount",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "CommunityOwnedDate",
                "ClosedDate",
                "LastEditDate",
                "LastActivityDate",
                "Score",
                "LastEditorDisplayName",
                "LastEditorUserId",
            ],
            inplace=True,
        )

        comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        comments = clean_datetime(comments, "CreationDate")
        badges = clean_datetime(badges, "Date")
        postLinks = clean_datetime(postLinks, "CreationDate")
        postHistory = clean_datetime(postHistory, "CreationDate")
        votes = clean_datetime(votes, "CreationDate")
        users = clean_datetime(users, "CreationDate")
        posts = clean_datetime(posts, "CreationDate")

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

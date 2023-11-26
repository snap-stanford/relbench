import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.stack_exchange import UserContributionTask, QuestionPopularityTask
from relbench.utils import unzip_processor, to_unix_time

class StackExchangeDataset(RelBenchDataset):
    name = "stack_exchange"
    val_timestamp = pd.Timestamp("2019-09-12")
    test_timestamp = pd.Timestamp("2021-09-12")
    task_cls_list = [UserContributionTask, QuestionPopularityTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = 'https://relbench.stanford.edu/data/relbench-forum-raw.zip'
        
        path = pooch.retrieve(
            url,
            known_hash=None,
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, 'raw')
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

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
            ],
            inplace=True,
        )

        comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        ## change time column to unix time
        comments["CreationDate"] = to_unix_time(comments["CreationDate"])
        badges["Date"] = to_unix_time(badges["Date"])
        postLinks["CreationDate"] = to_unix_time(postLinks["CreationDate"])

        postHistory["CreationDate"] = to_unix_time(postHistory["CreationDate"])
        votes["CreationDate"] = to_unix_time(votes["CreationDate"])
        posts["CreationDate"] = to_unix_time(posts["CreationDate"])
        users["CreationDate"] = to_unix_time(users["CreationDate"])

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
                "LastEditorUserId": "users",
                "ParentId": "posts",  # notice the self-reference
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        return Database(tables)
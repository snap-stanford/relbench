from __future__ import annotations

import os

import pandas as pd
import pooch
from torch_frame import stype

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.stack import (
    PostPostRelatedTask,
    PostVotesTask,
    UserBadgeTask,
    UserEngagementTask,
    UserPostCommentTask,
)
from relbench.utils import clean_datetime, unzip_processor


class StackDataset(RelBenchDataset):
    name = "rel-stack"
    # 3 months gap
    val_timestamp = pd.Timestamp("2020-10-01")
    test_timestamp = pd.Timestamp("2021-01-01")
    max_eval_time_frames = 1
    task_cls_list = [
        UserEngagementTask,
        PostVotesTask,
        UserBadgeTask,
        UserPostCommentTask,
        PostPostRelatedTask,
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
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
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

    @property
    def col_to_stype_dict(self) -> dict[str, dict[str, stype]]:
        return {
            "postLinks": {
                "Id": stype.numerical,
                "RelatedPostId": stype.numerical,
                "PostId": stype.numerical,
                "LinkTypeId": stype.numerical,
                "CreationDate": stype.timestamp,
            },
            "posts": {
                "Id": stype.numerical,
                "PostTypeId": stype.numerical,
                "AcceptedAnswerId": stype.numerical,
                "ParentId": stype.numerical,
                "CreationDate": stype.timestamp,
                "Body": stype.text_embedded,
                "OwnerUserId": stype.numerical,
                # "LastEditorUserId": stype.numerical,
                # Uninformative text column
                # "LastEditorDisplayName": stype.text_embedded,
                "Title": stype.text_embedded,
                "Tags": stype.text_embedded,
            },
            "users": {
                "Id": stype.numerical,
                "AccountId": stype.numerical,
                "CreationDate": stype.timestamp,
                # Uninformative text column
                # "DisplayName": stype.text_embedded,
                # "Location": stype.text_embedded,
                "AboutMe": stype.text_embedded,
                # Uninformative text column
                # "WebsiteUrl": stype.text_embedded,
            },
            "votes": {
                "Id": stype.numerical,
                "PostId": stype.numerical,
                "VoteTypeId": stype.numerical,
                "UserId": stype.numerical,
                "CreationDate": stype.timestamp,
            },
            "comments": {
                "Id": stype.numerical,
                "PostId": stype.numerical,
                "Text": stype.text_embedded,
                "CreationDate": stype.timestamp,
                "UserId": stype.numerical,
                # Uninformative text column
                # "UserDisplayName": stype.text_embedded,
                # "ContentLicense": stype.text_embedded,
            },
            "badges": {
                "Id": stype.numerical,
                "UserId": stype.numerical,
                "Class": stype.categorical,
                # Uninformative text column
                # "Name": stype.text_embedded,
                "Date": stype.timestamp,
                "TagBased": stype.categorical,
            },
            "postHistory": {
                "Id": stype.numerical,
                "PostId": stype.numerical,
                "UserId": stype.numerical,
                "PostHistoryTypeId": stype.numerical,
                # Uninformative text column
                # "UserDisplayName": stype.text_embedded,
                "ContentLicense": stype.categorical,
                # Uninformative text column
                # "RevisionGUID": stype.text_embedded,
                "Text": stype.text_embedded,
                # Uninformative text column
                # "Comment": stype.text_embedded,
                "CreationDate": stype.timestamp,
            },
        }

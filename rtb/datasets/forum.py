import json
import os
import re

import duckdb
import pandas as pd
from tqdm.auto import tqdm

from rtb.data.table import SemanticType, Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset


class ForumDataset(Dataset):

    name = "rtb-forum"

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""
        ## needs to brainstorm a bit about meaningful tasks
        #return {"post_views_one_week": grant_five_years()}
        return
        
    def download(self, path: str | os.PathLike) -> None:

        """
        Download a file from an S3 bucket.
        Parameters:
        - path (str): Local path where the file should be saved.
        Returns:
        None
        
        file_key = f"{self.root}/{self.name}"
        bucket_name = 'XXX' ## TBD
        region_name='us-west-2' ## TBD
        # Create an S3 client
        s3 = boto3.client('s3', region_name=region_name)
        # Download the file
        s3.download_file(bucket_name, file_key, path)   
        """
        pass

    def process(self) -> Database:
        r"""Process the raw files into a database."""
        path = f"{self.root}/{self.name}/raw/"
        badges = pd.read_csv(os.path.join(path, "badges.csv"))
        comments = pd.read_csv(os.path.join(path, "comments.csv"))
        postHistory = pd.read_csv(os.path.join(path, "postHistory.csv"))
        postLinks = pd.read_csv(os.path.join(path, "postLinks.csv"))
        posts = pd.read_csv(os.path.join(path, "posts.csv"))
        tags = pd.read_csv(os.path.join(path, "tags.csv"))
        users = pd.read_csv(os.path.join(path, "users.csv"))
        votes = pd.read_csv(os.path.join(path, "votes.csv"))

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            feat_cols={
                "Score": SemanticType.NUMERICAL,
                "Text": SemanticType.TEXT,
                "UserDisplayName": SemanticType.TEXT,
                "CreationDate": SemanticType.TIME
            },
            fkeys={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            feat_cols={
                "Name": SemanticType.TEXT,
                "Date": SemanticType.TIME,
            },
            fkeys={
                "UserId": "users",
            },
            pkey="Id",
            time_col="Date",
        )

        ## for tags, should we remove the count, excerptPostId and WikiPostId, for potential time leakage?
        
        tables["tags"] = Table(
            df=pd.DataFrame(tags),
            feat_cols={
                "Count": SemanticType.NUMERICAL,
            },
            fkeys={
                "ExcerptPostId": "posts", ## is this allowed? two foreign keys into the same primary
                "WikiPostId": "posts",
            },
            pkey="Id",
            time_col=None,
        )


        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            feat_cols={
                "LinkTypeId": SemanticType.CATEGORICAL,
            },
            fkeys={
                "PostId": "posts", 
                "RelatedPostId": "posts", ## is this allowed? two foreign keys into the same primary
            },
            pkey="Id",
            time_col="CreationDate",
        )


        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            feat_cols={
                "PostHistoryTypeId": SemanticType.CATEGORICAL,
                "RevisionGUID": SemanticType.TEXT,
                "Text": SemanticType.TEXT,
                "Comment": SemanticType.TEXT,
                "UserDisplayName": SemanticType.TEXT,
            },
            fkeys={
                "PostId": "posts",
                "UserId": "users"
            },
            pkey="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            feat_cols={
                "VoteTypeId": SemanticType.CATEGORICAL,
                "BountyAmount": SemanticType.NUMERICAL,
            },
            fkeys={
                "PostId": "posts",
                "UserId": "users"
            },
            pkey="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            feat_cols={
                "Reputation": SemanticType.NUMERICAL,
                "DisplayName": SemanticType.TEXT,
                "LastAccessDate": SemanticType.TIME, ## should this be used as time_col?
                "WebsiteUrl": SemanticType.TEXT,
                "Location": SemanticType.TEXT,
                "AboutMe": SemanticType.TEXT,
                "Views": SemanticType.NUMERICAL, ## is this time leakage?
                "UpVotes": SemanticType.NUMERICAL, ## is this time leakage?
                "DownVotes": SemanticType.NUMERICAL, ## is this time leakage?
                "AccountId": SemanticType.NUMERICAL,
                "Age": SemanticType.NUMERICAL, 
                "ProfileImageUrl": SemanticType.TEXT,
            },
            fkeys={},
            pkey="Id",
            time_col=None,
        )


        tables["posts"] = Table(
            df=pd.DataFrame(users),
            feat_cols={
                "CreaionDate": SemanticType.TIME,
                "Score": SemanticType.NUMERICAL,
                "ViewCount": SemanticType.NUMERICAL,
                "Body": SemanticType.TEXT,
                "LasActivityDate": SemanticType.TIME,## should this be used as time_col?
                "Title": SemanticType.TEXT,
                "Tags": SemanticType.TEXT, ## should we refer it back to the tags table? currently it is tagName, not tag id, also, if refer, there will be multiple tag IDs.
                "AnswerCount": SemanticType.NUMERICAL,
                "CommentCount": SemanticType.NUMERICAL,
                "FavoriteCount": SemanticType.NUMERICAL,
                "LastEditDate": SemanticType.TIME,
                "CommunityOwnedDate": SemanticType.TIME,
                "ClosedDate": SemanticType.TIME,
                "OwnerDisplayName": SemanticType.TEXT,
                "LastEditorDisplayName": SemanticType.TEXT,
                "ParentId": SemanticType.TEXT, ## should this be used? self-reference 
                "PostTypeId": SemanticType.CATEGORICAL,                
            },
            fkeys={
                "OwnerUserId": "users",
                "LastEditorUserId": "users",
            },
            pkey="Id",
            time_col="CreationDate",
        )
        
        return Database(tables)

    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        train_cutoff_time = 2008
        val_cutoff_time = 2013
        return train_cutoff_time, val_cutoff_time
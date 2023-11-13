import json
import os
import re

from typing import Dict, Union, Tuple
import duckdb
import pandas as pd
from tqdm.auto import tqdm

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset
from rtb.utils import to_unix_time


class ForumDataset(Dataset):
    name = "rtb-forum"

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""
        ## needs to brainstorm a bit about meaningful tasks
        # return {"post_views_one_week": grant_five_years()}
        return

    def download(self, path: Union[str, os.PathLike]) -> None:
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
            fkeys={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkeys={
                "UserId": "users",
            },
            pkey="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkeys={
                "PostId": "posts",
                "RelatedPostId": "posts",  ## is this allowed? two foreign keys into the same primary
            },
            pkey="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkeys={"PostId": "posts", "UserId": "users"},
            pkey="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkeys={"PostId": "posts", "UserId": "users"},
            pkey="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkeys={},
            pkey="Id",
            time_col=None,
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkeys={
                "OwnerUserId": "users",
                "LastEditorUserId": "users",
                "ParentId": "posts",  # notice the self-reference
            },
            pkey="Id",
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

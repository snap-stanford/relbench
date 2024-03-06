import os
from pathlib import Path
from typing import Tuple

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
from relbench.utils import unzip_processor


class StackExDataset(RelBenchDataset):
    name = "rel-stackex"
    url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
    known_hash = "ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f"
    inf_file_line = f"{url} = {known_hash}"
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
        use_db_cache: bool = True,
        keep_raw_csv: bool = False,
    ):
        self.name = f"{self.name}"
        self.local_db_cache_path = os.path.join(
            pooch.os_cache(self.name), self.db_dir, "raw"
        )
        self.inf_file_path = os.path.join(self.local_db_cache_path, "db.inf")
        self.use_db_cache = use_db_cache
        self.keep_raw_csv = keep_raw_csv  # don't delete imtermediate csv files

        super().__init__(process=process)

    def create_db(self) -> Tuple[Database, str]:
        r"""Process the raw files into a database."""
        path = pooch.retrieve(
            self.url,
            known_hash=self.known_hash,
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

        return Database(tables), path

    def check_db(self) -> bool:
        r"Check if local database is OK"
        if Path(self.inf_file_path).exists():
            with open(self.inf_file_path) as f:
                # check if inf_file contains matching url and hash of zip file
                return f.readline().strip() == self.inf_file_line
        return False

    def clear_local_db(self, db_path: str):
        print("Removing db cache files")
        for table_path in Path(db_path).glob("*.parquet"):
            os.remove(str(table_path))
        os.remove(os.path.join(db_path, "db.inf"))

    def clear_csv(self, csv_path: str):
        print("Removing raw csv files")
        for table_path in Path(csv_path).glob("*.csv"):
            os.remove(str(table_path))

    def make_db(self) -> Database:
        r"""Process the raw files into a database or load cached database."""
        db_path = self.local_db_cache_path
        if self.use_db_cache and self.check_db():
            print(f"Loading db from {db_path}")
            db = Database.load(db_path)
        else:
            db, csv_path = self.create_db()
            if self.use_db_cache:
                if not self.keep_raw_csv:
                    # raw csv files are not needed when db_cache is used
                    self.clear_csv(csv_path)

                print(f"saving Database object to {db_path}")
                db.save(db_path)

                # add db.inf file to the db_path with url and known_hash
                with open(self.inf_file_path, "w") as f:
                    f.write(f"{self.inf_file_line}\n")
            elif self.check_db():
                # local cache database files are not needed in this case
                self.clear_local_db(db_path)

        return db

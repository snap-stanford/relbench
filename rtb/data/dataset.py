from collections import defaultdict
import os
from pathlib import Path
import boto3
import json

import rtb


class Dataset:
    r"""Base class for dataset.

    Includes database, tasks, downloading, pre-processing and unified splitting.

    task_fns are functions that take a Database and create a task. The input
    database to these functions is only one split. This ensures that the task
    table for a split only uses information available in that split."""

    splits: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

    # name of dataset, to be specified by subclass
    name: str

    # task name -> task function, to be specified by subclass
    task_fns: dict[str, callable[[rtb.data.database.Database], rtb.data.task.Task]]

    def __init__(self, root: str, task_names: list[str] = []) -> None:
        r"""Initializes the dataset.

        Args:
            root: root directory to store dataset.
            task_names: list of tasks to create.

        The Dataset class exposes the following attributes:
            db_splits: split name -> Database
            task_splits: task name -> split name -> Task
        """
        self.root = root
        # download
        path = f"{root}/{self.name}/raw"
        if not Path(f"{path}/done").exists():
            self.download(path)
            Path(f"{path}/done").touch()

        # process, standardize and split
        path = f"{root}/{name}/processed/db"
        if not Path(f"{path}/done").exists():
            db = self.standardize_db(self.process_db())

            # save database splits independently
            db_splits = self.split_db(db)
            for split, db in db_splits.items():
                db.save(f"{path}/{split}")
            Path(f"{path}/done").touch()

        # load database splits
        self.db_splits = {
            split: rtb.data.database.Database.load(f"{path}/{split}")
            for split in ["train", "val", "test"]
        }

        # create tasks for each split
        self.task_splits = defaultdict(dict)
        for task_name in task_names:
            for split in ["train", "val", "test"]:
                path = f"{root}/{name}/processed/tasks/{task_name}/{split}"

                # save task split independent of other splits and tasks
                if not Path(f"{path}/done").exists():
                    task = self.task_fns[task_name](self.db_splits[split])
                    task.save(path)
                    Path(f"{path}/done").touch()

                # load task split
                self.task_splits[task_name][split] = Task.load(path)

    def download(self, path: str | os.PathLike) -> None:
        """
        Download a file from an S3 bucket.

        Parameters:
        - path (str): Local path where the file should be saved.

        Returns:
        None
        """
        
        file_key = f"{self.root}/{self.name}"
        bucket_name = 'XXX' ## TBD
        region_name='us-west-2' ## TBD
        
        # Create an S3 client
        s3 = boto3.client('s3', region_name=region_name)

        # Download the file
        s3.download_file(bucket_name, file_key, path)


    def process_db(self) -> rtb.data.database.Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""
        
        '''
        following this structure currently...
        info.json:
        
        {
            "dim_tables": ["customer", "brand", "product"],
            "fact_tables": ["review"],
            "schema": {
                "customer": {
                    "name": "text"
                },
                "product": {
                    "brand": "text",
                    "category": "text",
                    "description": "text",
                    "title": "text",
                    "image": "image",
                    "features": "text",
                    "price": "numerical"
                },
                "review": {
                    "creation_time": "timestamp",
                    "customer_id": "foreign_key[reviewer]",
                    "product_id": "foreign_key[product]",
                    "rating": "numerical",
                    "verified": "categorical",
                    "review_text": "text",
                    "summary": "text",
                    "vote": "numerical"
                }
            }
        }
        
        '''
        
        ## read info json file
        info = json.read(f"{self.root}/{self.name}/processed/info.json")
        
        table2rtb_table = {}
        
        for table_name in info["schema"].keys():
            df = pd.read_csv(os.path.join(f"{root}/{name}/processed/db", table_name, ".csv"))
            if table_name in info["dim_tables"]:
                pkey = table_name
            else:
                pkey = None
            time_col = None
            fkeys = []
            feat_cols = []
            for row_key, row_val in info["schema"][table_name].items():
                if row_val == 'timestamp':
                    time_col = row_key
                elif 'foreign_key' in row_val:
                    fkeys.append((row_key, row_val.split("[")[1].rstrip("]")))
                else:
                    feat_cols.append(row_key)
            table2rtb_table[table_name] = rtb.data.table.Table(df, feat_cols, fkeys, pkey, time_col)
                 
        return rtb.data.database.Database(table2rtb_table)


    def standardize_db(
        self, db: rtb.data.database.Database
    ) -> rtb.data.database.Database:
        r"""
        - Add primary key column if not present.
        - Re-index primary key column with 0-indexed ints, if required.
        """

        raise NotImplementedError

    def split_db(
        self, db: rtb.data.database.Database
    ) -> dict[str, rtb.data.database.Database]:
        r"""Splits the database into train, val, and test splits."""

        assert sum(self.splits.values()) == 1.0

        # get time stamps for splits
        self.val_split_time = db.time_of_split(splits["train"])
        self.test_split_time = db.time_of_split(splits["train"] + splits["val"])

        # split the database
        db_train, db_val_test = db.split_at(self.val_split_time)
        db_val, db_test = val_test.split_at(self.test_split_time)

        return {"train": db_train, "val": db_val, "test": db_test}

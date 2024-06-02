import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pooch

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.task_base import BaseTask
from relbench.utils import unzip_processor


class Dataset:
    def __init__(
        self,
        db: Database,
        train_start_timestamp: Optional[pd.Timestamp],
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        max_eval_time_frames: int,
        task_cls_list: List[Type[BaseTask]],
    ) -> None:
        r"""Class holding database and task table construction logic.

        Args:
            db (Database): The database object.
            train_start_timestamp (pd.Timestamp, optional): If specified, we create
                train table after the specified time.
            val_timestamp (pd.Timestamp): The first timestamp for making val table.
            test_timestamp (pd.Timestamp): The first timestamp for making test table.
            max_eval_time_frames (int): The maximum number of unique timestamps used to build test and val tables.
            task_cls_list (List[Type[BaseTask]]): A list of allowed tasks for this database.

        """
        self._full_db = db
        self.train_start_timestamp = train_start_timestamp
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.max_eval_time_frames = max_eval_time_frames
        self.task_cls_dict = {task_cls.name: task_cls for task_cls in task_cls_list}

        self.db = db.upto(test_timestamp)

        self.validate_and_correct_db()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def task_names(self) -> List[str]:
        return list(self.task_cls_dict.keys())

    def get_task(self, task_name: str, *args, **kwargs) -> BaseTask:
        if task_name not in self.task_cls_dict:
            raise ValueError(
                f"{self.__class__.name} does not support the task {task_name}."
                f"Please choose from {self.task_names}."
            )
        return self.task_cls_dict[task_name](self, *args, **kwargs)

    def validate_and_correct_db(self):
        r"""Validate and correct input db in-place."""
        # Validate that all primary keys are consecutively index.

        for table_name, table in self.db.table_dict.items():
            if table.pkey_col is not None:
                ser = table.df[table.pkey_col]
                if not (ser.values == np.arange(len(ser))).all():
                    raise RuntimeError(
                        f"The primary key column {table.pkey_col} of table "
                        f"{table_name} is not consecutively index."
                    )

        # Discard any foreign keys that are larger than primary key table as
        # dangling foreign keys (represented as None).
        for table_name, table in self.db.table_dict.items():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                num_pkeys = len(self.db.table_dict[pkey_table_name])
                mask = table.df[fkey_col] >= num_pkeys
                if mask.any():
                    table.df.loc[mask, fkey_col] = None


class RelBenchDataset(Dataset):
    name: str
    train_start_timestamp: Optional[pd.Timestamp] = None
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    task_cls_list: List[Type[BaseTask]]

    db_dir: str = "db"

    def __init__(self, *, process: bool = False) -> None:
        if process:
            print("making Database object from raw files...")
            tic = time.time()
            db = self.make_db()
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            print("reindexing pkeys and fkeys...")
            tic = time.time()
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            db_path = pooch.os_cache("relbench") / self.name / self.db_dir
            print(f"caching Database object to {db_path}...")
            tic = time.time()
            db.save(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")
            print(f"use process=False to load from cache.")

        else:
            db_path = _pooch.fetch(
                f"{self.name}/{self.db_dir}.zip",
                processor=unzip_processor,
                progressbar=True,
            )
            print(f"loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        super().__init__(
            db,
            self.train_start_timestamp,
            self.val_timestamp,
            self.test_timestamp,
            self.max_eval_time_frames,
            self.task_cls_list,
        )

    def make_db(self) -> Database:
        raise NotImplementedError

    def pack_db(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / self.db_dir
            print(f"saving Database object to {db_path}...")
            tic = time.time()
            self._full_db.save(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            print("making zip archive for db...")
            tic = time.time()
            zip_path = Path(root) / self.name / self.db_dir
            zip_path = shutil.make_archive(zip_path, "zip", db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        with open(zip_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"upload: {zip_path}")
        print(f"sha256: {sha256}")

        return f"{self.name}/{self.db_dir}.zip", sha256

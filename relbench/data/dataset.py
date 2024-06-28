import hashlib
import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pooch

from relbench import DOWNLOAD_REGISTRY
from relbench.data.database import Database
from relbench.data.task_base import BaseTask
from relbench.utils import unzip_processor


class Dataset:
    def __init__(
        self,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        db_path: str = None,
        max_eval_time_frames: int = 1,
    ) -> None:
        r"""Class holding database and task table construction logic.

        Args:
            db (Database): The database object.
            val_timestamp (pd.Timestamp): The first timestamp for making val table.
            test_timestamp (pd.Timestamp): The first timestamp for making test table.
            max_eval_time_frames (int): The maximum number of unique timestamps used to build test and val tables.

        """
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp

        if db_path is None:
            db_path = tempfile.mkdtemp()
        self.db_path = db_path

        self.max_eval_time_frames = max_eval_time_frames

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

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

    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp=True) -> Database:
        if not Path(self.db_path).exists() or not any(Path(self.db_path).iterdir()):
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

            print(f"caching Database object to {self.db_path}...")
            tic = time.time()
            db.save(self.db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")
            print(f"use process=False to load from cache.")

        else:
            print(f"loading Database object from {self.db_path}...")
            tic = time.time()
            db = Database.load(self.db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        return db

    def make_db(self) -> Database:
        raise NotImplementedError

    # TODO: move out of here
    def pack_db(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "db"
            print(f"saving Database object to {db_path}...")
            tic = time.time()
            self._full_db.save(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            print("making zip archive for db...")
            tic = time.time()
            zip_path = Path(root) / self.name / "db"
            zip_path = shutil.make_archive(zip_path, "zip", db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        with open(zip_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"upload: {zip_path}")
        print(f"sha256: {sha256}")

        return f"{self.name}/db.zip", sha256

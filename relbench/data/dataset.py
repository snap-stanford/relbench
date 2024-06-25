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
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        cache_dir: str | None = None,
    ) -> None:
        r"""Class holding database and task table construction logic.

        Args:
            db (Database): The database object.
            val_timestamp (pd.Timestamp): The first timestamp for making val table.
            test_timestamp (pd.Timestamp): The first timestamp for making test table.
        """
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + f"  val_timestamp={self.val_timestamp},\n"
            + f"  test_timestamp={self.test_timestamp},\n"
            + f"  cache_dir={self.cache_dir},\n"
            + ")"
        )

    def get_db(self, upto_test_timestamp=True) -> Database:
        if self.cache_dir is None:
            db = self._make_db()
        else:
            db_path = f"{self.cache_dir}/db"
            if not Path(db_path).exists():
                db = self._make_db()
                db.reindex_pkeys_and_fkeys()
                db.save(db_path)
            else:
                db = Database.load(db_path)

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        return db

    def _make_db(self) -> Database:
        raise NotImplementedError


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

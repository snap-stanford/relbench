import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import pooch

from relbench import _pooch
from relbench.data.database import Database
from relbench.data.task import Task
from relbench.utils import unzip_processor


class Dataset:
    def __init__(
        self,
        db: Database,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        task_cls_list: List[Type[Task]],
    ) -> None:
        self._full_db = db
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.task_cls_dict = {task_cls.name: task_cls for task_cls in task_cls_list}

        self.db = db.upto(test_timestamp)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def task_names(self) -> List[str]:
        return list(self.task_cls_dict.keys())

    def get_task(self, task_name: str, *args, **kwargs) -> Task:
        return self.task_cls_dict[task_name](self, *args, **kwargs)


class RelBenchDataset(Dataset):
    name: str
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    task_cls_list: List[Type[Task]]

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

        else:
            db_path = _pooch.fetch(
                f"{self.name}/{self.db_dir}.zip",
                processor=unzip_processor,
                progressbar=True,
            )
            db = Database.load(db_path)

        super().__init__(
            db, self.val_timestamp, self.test_timestamp, self.task_cls_list
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

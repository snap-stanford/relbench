import os
import shutil
import time
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from rtb.data.database import Database
from rtb.data.table import Table
from rtb.data.task import Task
from rtb.utils import download_url, one_window_sampler, rolling_window_sampler, unzip


class Dataset:
    def __init__(
        self,
        db: Database,
        train_max_time: pd.Timestamp,
        val_max_time: pd.Timestamp,
        task_cls_dict: Dict[str, Task],
    ) -> None:
        self._db = db
        self.train_max_time = train_max_time
        self.val_max_time = val_max_time
        self.task_cls_dict = task_cls_dict

        self.db_min_time = db.min_time
        self.db_max_time = db.max_time

    @property
    @cache
    def db_upto_train(self) -> Database:
        return self._db.upto(self.train_max_time)

    @property
    @cache
    def db_upto_val(self) -> Database:
        return self._db.upto(self.val_max_time)

    @property
    @cache
    def db(self) -> Dataset:
        warnings.warn(
            "Accessing the full database is not recommended. Use with caution"
            "to prevent temporal leakage. If memory is not a concern, please use"
            "`db_upto_train` or `db_upto_val` instead."
        )
        return self._db

    def get_task(self, task_name: str) -> Task:
        return self.task_cls_dict[task_name](self._db)


class RelBenchDataset(Dataset):
    name: str
    task_cls_dict: Dict[str, type[Task]]

    raw_url_fmt: str = "http://relbench.stanford.edu/data/raw/{}.zip"
    processed_url_fmt: str = "http://relbench.stanford.edu/data/processed/{}.zip"

    raw_dir: str = "raw"
    processed_dir: str = "processed"
    db_dir: str = "db"
    split_times_file: str = "split_times.json"

    def __init__(
        root=Union[str, os.PathLike],
        *,
        download=False,
        process=False,
    ):
        root = Path(root)

        processed_path = root / self.name / self.processed_dir
        processed_path.mkdir(parents=True, exist_ok=True)

        if process:
            raw_path = root / self.name / self.raw_dir
            raw_path.mkdir(parents=True, exist_ok=True)

            if download or not (raw_path / "done").exists():
                print("downloading raw files...")
                tic = time.time()
                raw_url = self.raw_url_fmt.format(self.name)
                download_and_extract(raw_url, raw_path)
                toc = time.time()
                print(f"downloading raw files took {toc - tic:.2f} seconds.")

                (raw_path / "done").touch()

            print("processing db...")
            tic = time.time()
            db = self.process_db(raw_path)
            toc = time.time()
            print(f"processing db took {toc - tic:.2f} seconds.")

            print("reindexing pkeys and fkeys...")
            tic = time.time()
            db = self.reindex_pkeys_and_fkeys(db)
            toc = time.time()
            print(f"reindexing pkeys and fkeys took {toc - tic:.2f} seconds.")

            print("saving db...")
            tic = time.time()
            db.save(processed_path / self.db_dir)
            toc = time.time()
            print(f"saving db took {toc - tic:.2f} seconds.")

            print("saving split times...")
            tic = time.time()
            train_max_time, val_max_time = self.get_split_times(db)
            with open(processed_path / self.split_times_file, "w") as f:
                json.dump(
                    {
                        "train_max_time": str(train_max_time),
                        "val_max_time": str(val_max_time),
                    },
                    f,
                )
            toc = time.time()
            print(f"saving split times took {toc - tic:.2f} seconds.")

            (processed_path / "done").touch()

        else:
            if download or not (processed_path / "done").exists():
                print("downloading processed files...")
                tic = time.time()
                processed_url = self.processed_url_fmt.format(self.name)
                download_and_extract(processed_url, processed_path)
                toc = time.time()
                print(f"downloading processed files took {toc - tic:.2f} seconds.")

                (processed_path / "done").touch()

            db = Database.load(processed_path)

        train_max_time, val_max_time = self.get_split_times(db)

        super().__init__(
            db=db,
            train_max_time=train_max_time,
            val_max_time=val_max_time,
            task_cls_dict=self.task_cls_dict,
        )

    def process_db(self, raw_path: Union[str, os.PathLike]) -> Database:
        raise NotImplementedError

    def get_split_times(self, db: Database) -> Tuple[pd.Timestamp, pd.Timestamp]:
        train_max_time, val_max_time = db.split_times([0.8, 0.9])
        return train_max_time, val_max_time

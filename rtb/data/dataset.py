import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from rtb.data.database import Database
from rtb.data.task import Task
from rtb.utils import download_and_extract


class Dataset:
    def __init__(
        self,
        db: Database,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        task_cls_list: List[Type[Task]],
    ) -> None:
        self._db = db
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.task_cls_list = task_cls_list

        self.input_db = db.upto(test_timestamp)
        self.min_timestamp = db.min_timestamp
        self.max_timestamp = db.max_timestamp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def task_names(self) -> List[str]:
        return [task_cls.name for task_cls in self.task_cls_list]

    def get_task(self, task_name: str) -> Task:
        for task_cls in self.task_cls_list:
            if task_cls.name == task_name:
                return task_cls(self)
        raise ValueError(f"Unknown task {task_name} for dataset {self}.")


class BenchmarkDataset(Dataset):
    name: str
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    task_cls_list: List[Type[Task]]

    raw_url_fmt: str = "http://relbench.stanford.edu/data/raw/{}.zip"
    processed_url_fmt: str = "http://relbench.stanford.edu/data/processed/{}.zip"

    raw_dir: str = "raw"
    processed_dir: str = "processed"

    def __init__(
        root=Union[str, os.PathLike],
        *,
        download=False,
        process=False,
    ):
        root = Path(root)

        processed_path = root / self.name / self.processed_dir

        if process:
            raw_path = root / self.name / self.raw_dir
            raw_path.mkdir(parents=True, exist_ok=True)

            if download or not (raw_path / "done").exists():
                # delete to avoid corruption
                shutil.rmtree(raw_path, ignore_errors=True)
                raw_path.mkdir(parents=True)

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
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"reindexing pkeys and fkeys took {toc - tic:.2f} seconds.")

            # delete to avoid corruption
            shutil.rmtree(processed_path, ignore_errors=True)
            processed_path.mkdir(parents=True)

            print("saving db...")
            tic = time.time()
            db.save(processed_path)
            toc = time.time()
            print(f"saving db took {toc - tic:.2f} seconds.")

            (processed_path / "done").touch()

        else:
            if download or not (processed_path / "done").exists():
                # delete to avoid corruption
                shutil.rmtree(processed_path, ignore_errors=True)
                processed_path.mkdir(parents=True)

                print("downloading processed files...")
                tic = time.time()
                processed_url = self.processed_url_fmt.format(self.name)
                download_and_extract(processed_url, processed_path)
                toc = time.time()
                print(f"downloading processed files took {toc - tic:.2f} seconds.")

                (processed_path / "done").touch()

            print("loading db...")
            tic = time.time()
            db = Database.load(processed_path)
            toc = time.time()
            print(f"loading db took {toc - tic:.2f} seconds.")

        super().__init__(
            db, self.val_timestamp, self.test_timestamp, self.task_cls_list
        )

    def process_db(self, raw_path: Union[str, os.PathLike]) -> Database:
        raise NotImplementedError

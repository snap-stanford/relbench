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
        self.task_cls_dict = {task_cls.name: task_cls for task_cls in task_cls_list}

        self.input_db = db.upto(test_timestamp)
        self.min_timestamp = db.min_timestamp
        self.max_timestamp = db.max_timestamp

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

    processed_url_fmt: str = "http://relbench.stanford.edu/data/{}/db/processed.zip"

    db_dir: str = "db"
    raw_dir: str = "raw"
    processed_dir: str = "processed"

    def __init__(
        self,
        root=Union[str, os.PathLike],
        *,
        download=False,
        process=False,
    ):
        root = Path(root)
        self.root = root

        processed_path = root / self.name / self.db_dir / self.processed_dir

        if process:
            raw_path = root / self.name / self.db_dir / self.raw_dir
            raw_path.mkdir(parents=True, exist_ok=True)

            if download or not (raw_path / "done").exists():
                # delete to avoid corruption
                shutil.rmtree(raw_path, ignore_errors=True)
                raw_path.mkdir(parents=True)

                print(f"downloading raw files into {raw_path}...")
                tic = time.time()
                self.download_raw(raw_path)
                toc = time.time()
                print(f"downloading raw files took {toc - tic:.2f} seconds.")

                (raw_path / "done").touch()

            print("processing db...")
            tic = time.time()
            db = self.make_db(raw_path)
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

            print("creating zip archive...")
            tic = time.time()
            shutil.make_archive(processed_path, "zip", processed_path)
            toc = time.time()
            print(f"creating zip archive took {toc - tic:.2f} seconds.")

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

    def download_raw_db(self, raw_path: Union[str, os.PathLike]) -> None:
        raise NotImplementedError

    def make_db(self, raw_path: Union[str, os.PathLike]) -> Database:
        raise NotImplementedError

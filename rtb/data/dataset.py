import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from rtb.data import Database, Task
from rtb.utils import download_and_extract


class Dataset:
    def __init__(
        self,
        db: Database,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        task_cls_dict: Dict[str, Task],
    ) -> None:
        self._db = db
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.task_cls_dict = task_cls_dict

        self.input_db = db.upto(test_timestamp)
        self.min_timestamp = db.min_timestamp
        self.max_timestamp = db.max_timestamp

    def get_task(self, task_name: str, timedelta: pd.Timedelta) -> Task:
        return self.task_cls_dict[task_name](self._db, timedelta)


class BenchmarkDataset(Dataset):
    name: str
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    task_cls_dict: Dict[str, type[Task]]

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
            db = self.reindex_pkeys_and_fkeys(db)
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
            db, self.val_timestamp, self.test_timestamp, self.task_cls_dict
        )

    def process_db(self, raw_path: Union[str, os.PathLike]) -> Database:
        raise NotImplementedError

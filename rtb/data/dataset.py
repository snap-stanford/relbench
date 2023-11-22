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


class RawDataset:
    r"""Base class for dataset. A dataset includes a database and tasks defined
    on it."""

    # name of dataset, to be specified by subclass
    name: str

    def __init__(
        self, root: Union[str, os.PathLike], process=False, download=False
    ) -> None:
        r"""Initializes the dataset.

        process=True wil force pre-processing data from raw files.
        download=True will force downloading raw files if process=True, or
        processed files if process=False.

        By default, tries to download (if required) and use already processed data.
        """

        self.root = root

        process_path = os.path.join(root, self.name, "processed", "db")
        os.makedirs(process_path, exist_ok=True)

        if process:
            # download
            # raw_path = f"{root}/{self.name}/raw"
            raw_path = os.path.join(root, self.name, "raw")
            os.makedirs(raw_path, exist_ok=True)
            if download or not Path(f"{raw_path}/done").exists():
                self.download_raw(os.path.join(root, self.name))
                Path(f"{raw_path}/done").touch()

            # delete processed db dir if exists to avoid possibility of
            # corruption
            shutil.rmtree(process_path, ignore_errors=True)

            # process db
            print("processing db...")
            tic = time.time()
            db = self.process()
            toc = time.time()
            print(f"processing db took {toc - tic:.2f} seconds.")

            # standardize db
            print("reindexing pkeys and fkeys...")
            tic = time.time()
            # db = self.reindex_pkeys_and_fkeys(db)
            # TODO: also sort by time
            toc = time.time()
            print(f"reindexing pkeys and fkeys took {toc - tic:.2f} seconds.")

            # process and reindex_pkeys_and_fkeys are separate because
            # process() is implemented by each subclass, but
            # reindex_pkeys_and_fkeys() is common to all subclasses

            db.save(process_path)
            Path(f"{process_path}/done").touch()

            self._db = db

        else:
            # download
            if download or not Path(f"{process_path}/done").exists():
                self.download_processed(os.path.join(root, self.name, "processed"))
                Path(f"{process_path}/done").touch()

            # load database
            self._db = Database.load(process_path)

        # we want to keep the database private, because it also contains
        # test information

        self.min_time, self.max_time = self._db.get_time_range()
        self.train_max_time, self.val_max_time = self.get_cutoff_times()

        self.tasks = self.get_tasks()

    def download_raw(self, path: Union[str, os.PathLike]) -> None:
        """Downloads the raw data to the path directory. For our
        datasets, we will download from stanford URL, but we should keep it as
        a function for Dataset subclasses to override if required."""

        url = f"http://ogb-data.stanford.edu/data/rtb/{self.name}-raw.zip"
        download_path = download_url(url, path)
        unzip(download_path, path)

    def download_processed(self, path: Union[str, os.PathLike]) -> None:
        """Downloads the processed data to the path directory. For our
        datasets, we will download from stanford URL, but we should keep it as
        a function for Dataset subclasses to override if required."""
        url = f"http://ogb-data.stanford.edu/data/rtb/{self.name}-processed.zip"
        download_path = download_url(url, path)
        unzip(download_path, path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  tables={sorted(list(self._db.tables.keys()))},\n"
            f"  tasks={sorted(list(self.tasks.keys()))},\n"
            f"  min_time={self.min_time},\n"
            f"  max_time={self.max_time},\n"
            f"  train_max_time={self.train_max_time},\n"
            f"  val_max_time={self.val_max_time},\n"
            f")"
        )

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset. To be implemented
        by subclass."""

        raise NotImplementedError

    def get_cutoff_times(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        train_max_time = self.min_time + 0.8 * (self.max_time - self.min_time)
        val_max_time = self.min_time + 0.9 * (self.max_time - self.min_time)
        return train_max_time, val_max_time

    def process(self) -> Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""

        raise NotImplementedError

    @property
    def db(self) -> Database:
        r"""The full database. Use with caution to prevent temporal leakage."""
        return self._db

    @property
    def db_train(self) -> Database:
        return self._db.time_cutoff(self.train_max_time)

    @property
    def db_val(self) -> Database:
        return self._db.time_cutoff(self.val_max_time)

    def make_train_table(
        self,
        task_name: str,
        window_size: Optional[int] = None,
        time_window_df: Optional[pd.DataFrame] = None,
    ) -> Table:
        """Returns the train table for a task.

        User can either provide the window_size and get the train table
        generated by our default sampler, or explicitly provide the
        time_window_df obtained by their sampling strategy."""

        if time_window_df is None:
            if window_size is None:
                window_size = self.tasks[task_name].window_sizes[0]

            # default sampler
            time_window_df = rolling_window_sampler(
                self.min_time,
                self.train_max_time,
                window_size,
                stride=window_size,
            )

        task = self.tasks[task_name]
        return task.make_table(self.db_train, time_window_df)

    def make_val_table(
        self,
        task_name: str,
        window_size: Optional[int] = None,
        time_window_df: Optional[pd.DataFrame] = None,
    ) -> Table:
        r"""Returns the val table for a task.

        User can either provide the window_size and get the train table
        generated by our default sampler, or explicitly provide the
        time_window_df obtained by their sampling strategy."""

        if time_window_df is None:
            if window_size is None:
                window_size = self.tasks[task_name].window_sizes[0]

            # default sampler
            time_window_df = one_window_sampler(
                self.train_max_time,
                window_size,
            )

        task = self.tasks[task_name]
        return task.make_table(self.db_val, time_window_df)

    def make_test_table(
        self,
        task_name: str,
        window_size: Optional[int] = None,
    ) -> Table:
        r"""Returns the test table for a task."""
        if window_size is None:
            window_size = self.tasks[task_name].window_sizes[0]

        task = self.tasks[task_name]
        time_window_df = one_window_sampler(
            self.val_max_time,
            window_size,
        )
        table = task.make_table(self._db, time_window_df)

        # mask out non-input cols to prevent any info leakage
        table.df = table.df[task.input_cols]

        return table

    def get_stype_proposal(self) -> Dict[str, Dict[str, Any]]:
        r"""Propose stype for columns of a set of tables.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping table name into
                :obj:`col_to_stype` (mapping column names into inferred stypes)
        """
        from torch_frame.utils import infer_df_stype

        inferred_col_to_stype_dict = {}
        for table_name, table in self.db.tables.items():
            inferred_col_to_stype = infer_df_stype(table.df)

            # Temporarily removing time_col since StypeEncoder for
            # stype.timestamp is not yet supported.
            # TODO: Drop the removing logic once StypeEncoder is supported.
            # https://github.com/pyg-team/pytorch-frame/pull/225
            if table.time_col is not None:
                inferred_col_to_stype.pop(table.time_col)

            # Remove pkey, fkey columns since they will not be used as input
            # feature.
            if table.pkey_col is not None:
                inferred_col_to_stype.pop(table.pkey_col)
            for fkey in table.fkey_col_to_pkey_table.keys():
                inferred_col_to_stype.pop(fkey)

            inferred_col_to_stype_dict[table_name] = inferred_col_to_stype
        return inferred_col_to_stype_dict


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

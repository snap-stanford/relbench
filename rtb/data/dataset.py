import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rtb.data.database import Database
from rtb.data.table import Table
from rtb.data.task import Task
from rtb.utils import download_url, one_window_sampler, rolling_window_sampler, unzip


class Dataset:
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

        process_path = f"{root}/{self.name}/processed/db"

        if process:
            # download
            raw_path = f"{root}/{self.name}/raw"
            if download or not Path(f"{raw_path}/done").exists():
                self.download_raw(raw_path)
                Path(f"{raw_path}/done").touch()

            # delete processed db dir if exists to avoid possibility of corruption
            shutil.rmtree(process_path, ignore_errors=True)

            # process db
            print(f"processing db...")
            tic = time.time()
            db = self.process()
            toc = time.time()
            print(f"processing db took {toc - tic:.2f} seconds.")

            # standardize db
            print(f"reindexing pkeys and fkeys...")
            tic = time.time()
            db = self.reindex_pkeys_and_fkeys(db)
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
                self.download_processed(process_path)
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

        raise NotImplementedError

    def download_processed(self, path: Union[str, os.PathLike]) -> None:
        """Downloads the processed data to the path directory. For our
        datasets, we will download from stanford URL, but we should keep it as
        a function for Dataset subclasses to override if required."""
        url = f"http://ogb-data.stanford.edu/data/rtb/{self.name}.zip"
        # TODO: should be Path(f"{root}/{self.name}/") but that will break
        # current workflow for grant dataset
        # TODO: fix it together with a new zip file
        download_path = download_url(url, root)
        unzip(download_path, root)

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

    def download_raw(self, path: Union[str, os.PathLike]) -> None:
        r"""Downloads the raw data to the path directory. To be implemented by
        subclass."""

        raise NotImplementedError

    def process(self) -> Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""

        raise NotImplementedError

    def reindex_pkeys_and_fkeys(self, db: Database) -> None:
        r"""Mapping primary and foreign keys into indices according to
        the ordering in the primary key tables.

        Args:
            db (Database): The database object containing a set of tables.

        Returns:
            Database: Mapped database.
        """
        # Get pkey to idx mapping:
        index_map_dict: Dict[str, pd.Series] = {}
        for table_name, table in db.tables.items():
            if table.pkey_col is not None:
                ser = table.df[table.pkey_col]
                if ser.nunique() != len(ser):
                    raise RuntimeError(
                        f"The primary key '{table.pkey_col}' "
                        f"of table '{table_name}' contains "
                        "duplicated elements"
                    )
                arange_ser = pd.RangeIndex(len(ser)).astype("Int64")
                index_map_dict[table_name] = pd.Series(
                    index=ser,
                    data=arange_ser,
                    name="index",
                )
                table.df[table.pkey_col] = arange_ser

        # Replace fkey_col_to_pkey_table with indices.
        for table in db.tables.values():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                out = pd.merge(
                    table.df[fkey_col],
                    index_map_dict[pkey_table_name],
                    how="left",
                    left_on=fkey_col,
                    right_index=True,
                )
                table.df[fkey_col] = out["index"]

        return db

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

import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import Task
from rtb.utils import rolling_window_sampler, one_window_sampler, download_url, unzip


class Dataset:
    r"""Base class for dataset. A dataset includes a database and tasks defined
    on it."""

    # name of dataset, to be specified by subclass
    name: str

    def __init__(self, root: str | os.PathLike, process=False, download=False) -> None:
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
                self.download_raw(path)
                Path(f"{path}/done").touch()

            # delete processed db dir if exists to avoid possibility of corruption
            shutil.rmtree(path, ignore_errors=True)

            # process db
            print(f"processing db...")
            tic = time.time()
            db = self.process()
            toc = time.time()
            print(f"processing db took {toc - tic:.2f} seconds.")

            # standardize db
            print(f"standardizing db...")
            tic = time.time()
            db = self.standardize(db)
            toc = time.time()
            print(f"standardizing db took {toc - tic:.2f} seconds.")

            # process and standardize are separate because
            # process() is implemented by each subclass, but
            # standardize() is common to all subclasses

            db.save(path)
            Path(f"{path}/done").touch()

            self._db = db

        else:
            # download
            if download or not Path(f"{process_path}/done").exists():
                url = f"http://ogb-data.stanford.edu/data/rtb/{self.name}.zip"
                # TODO: should be Path(f"{root}/{self.name}/") but that will break
                # current workflow for grant dataset
                # TODO: fix it together with a new zip file
                download_path = download_url(url, root)
                unzip(download_path, root)

            # load database
            self._db = Database.load(process_path)

        # we want to keep the database private, because it also contains
        # test information

        self.min_time, self.max_time = self._db.get_time_range()
        self.train_max_time, self.val_max_time = self.get_cutoff_times()

        self.tasks = self.get_tasks()

    def __repr__(self):
        return (
            f"Dataset(\n"
            f"root={self.root},\n\n"
            f"min_time={self.min_time},\n\n"
            f"max_time={self.max_time},\n\n"
            f"train_max_time={self.train_max_time},\n\n"
            f"val_max_time={self.val_max_time},\n\n"
            f"tasks={self.tasks},\n\n"
            f"db_train={self.db_train}\n"
            f")"
        )

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset. To be implemented
        by subclass."""

        raise NotImplementedError

    def get_cutoff_times(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        train_max_time = self.min_time + 0.8 * (self.max_time - self.min_time)
        val_max_time = self.min_time + 0.9 * (self.max_time - self.min_time)
        return train_max_time, val_max_time

    def download_raw(self, path: str | os.PathLike) -> None:
        r"""Downloads the raw data to the path directory. To be implemented by
        subclass."""

        raise NotImplementedError

    def process(self) -> Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""

        raise NotImplementedError

    def standardize(self, db: Database) -> None:
        idx_dict = {}
        for name, table in db.tables.items():
            if table.pkey_col is None:
                continue
            s = table.df[table.pkey_col].reset_index(drop=True)
            # swap index and values, will be used to join later
            idx_dict[name] = pd.Series(
                s.index.values, index=s.values, name="__pkey_idx__"
            )
            # replace pkey col with index
            table.df[table.pkey_col] = s.index.values

        # replace fkeys with index of pkey table
        for name, table in db.tables.items():
            for fkey_col, pkey_table_name in table.fkeys.items():
                # inner join removes rows with fkeys that are not in pkey table
                # XXX: do we want this here? might hide preprocessing bugs
                table.df = table.df.join(
                    idx_dict[pkey_table_name], on=fkey_col, how="inner"
                )
                table.df.drop(columns=[fkey_col], inplace=True)
                table.df.rename(columns={"__pkey_idx__": fkey_col}, inplace=True)

        return db

    @property
    def db_train(self) -> Database:
        return self._db.time_cutoff(self.train_max_time)

    @property
    def db_val(self) -> Database:
        return self._db.time_cutoff(self.val_max_time)

    def make_train_table(
        self,
        task_name: str,
        window_size: int | None = None,
        time_window_df: pd.DataFrame | None = None,
    ) -> Table:
        """Returns the train table for a task.

        User can either provide the window_size and get the train table
        generated by our default sampler, or explicitly provide the
        time_window_df obtained by their sampling strategy."""

        if time_window_df is None:
            assert window_size is not None
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
        window_size: int | None = None,
        time_window_df: pd.DataFrame | None = None,
    ) -> Table:
        r"""Returns the val table for a task.

        User can either provide the window_size and get the train table
        generated by our default sampler, or explicitly provide the
        time_window_df obtained by their sampling strategy."""

        if time_window_df is None:
            assert window_size is not None
            # default sampler
            time_window_df = one_window_sampler(
                self.train_max_time,
                window_size,
            )

        task = self.tasks[task_name]
        return task.make_table(self.db_val, time_window_df)

    def make_test_table(self, task_name: str, window_size: int) -> Table:
        r"""Returns the test table for a task."""

        task = self.tasks[task_name]
        time_window_df = one_window_sampler(
            self.val_max_time,
            window_size,
        )
        table = task.make_table(self._db, time_window_df)

        # hide the label information
        df = table.df
        df.drop(columns=[task.target_col], inplace=True)
        table.df = df
        return table

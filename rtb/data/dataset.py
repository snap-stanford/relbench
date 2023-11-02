import os
from pathlib import Path

import rtb


class Dataset:
    r"""Base class for dataset. A dataset includes a database and tasks defined
    on it."""

    # name of dataset, to be specified by subclass
    name: str

    def __init__(self, root: str) -> None:
        r"""Initializes the dataset."""

        # download
        path = f"{root}/{self.name}/raw"
        if not Path(f"{path}/done").exists():
            self.download(path)
            Path(f"{path}/done").touch()

        path = f"{root}/{name}/processed/db"
        if not Path(f"{path}/done").exists():
            # process db
            db = self.process_db()

            # standardize db
            db = self.standardize_db()

            # process and standardize are separate because
            # process_db() is implemented by each subclass, but
            # standardize_db() is common to all subclasses

            db.save(path)

        # load database
        self._db = rtb.data.Database.load(path)

        # we want to keep the database private, because it also contains
        # test information

        self.min_time, self.max_time = self._db.get_time_range()
        self.train_cutoff_time, self.val_cutoff_time = self.get_cutoff_times()

        self.tasks = self.get_tasks()

    def get_tasks(self) -> dict[str, rtb.data.Task]:
        r"""Returns a list of tasks defined on the dataset. To be implemented
        by subclass."""

        raise NotImplementedError

    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        raise NotImplementedError

    def download(self, path: str | os.PathLike) -> None:
        r"""Downloads the raw data to the path directory. To be implemented by
        subclass."""

        raise NotImplementedError

    def process_db(self) -> rtb.data.database.Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""

        raise NotImplementedError

    def standardize_db(
        self, db: rtb.data.database.Database
    ) -> rtb.data.database.Database:
        r"""
        - Add primary key column if not present.
        - Re-index primary key column with 0-indexed ints, if required.
        - Can still keep the original pkey column as a feature column (e.g. email).
        """

        raise NotImplementedError

    def db_snapshot(self, time_stamp: int) -> rtb.data.database.Database:
        r"""Returns a database with all rows upto time_stamp (if table is
        temporal, otherwise all rows)."""

        assert time_stamp <= self.val_cutoff_time

        return self._db.time_cutoff(time_stamp)

    def get_test_table(self, task_name: str, time_window: int) -> rtb.data.table.Table:
        r"""Returns the test table for a task."""

        task = self.tasks[task_name]
        table = task.make_table(
            self._db,
            # just one time window into the future
            pd.DataFrame(
                {
                    "offset": [dset.train_cutoff_time],
                    "cutoff": [dset.train_cutoff_time + time_window],
                }
            ),
        )

        # hide the label information
        table.drop(columns=[task.target_col])

        return table

from collections import defaultdict
import os
from pathlib import Path

import rtb


class Dataset:
    r"""Base class for dataset.

    Includes database, tasks, downloading, pre-processing and unified splitting.

    task_fns are functions that take a Database and create a task. The input
    database to these functions is only one split. This ensures that the task
    table for a split only uses information available in that split."""

    splits: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

    # name of dataset, to be specified by subclass
    name: str

    # task name -> task function, to be specified by subclass
    task_fns: dict[str, callable[[rtb.data.database.Database], rtb.data.task.Task]]

    def __init__(self, root: str, task_names: list[str] = []) -> None:
        r"""Initializes the dataset.

        Args:
            root: root directory to store dataset.
            task_names: list of tasks to create.

        The Dataset class exposes the following attributes:
            db_splits: split name -> Database
            task_splits: task name -> split name -> Task
        """

        # download
        path = f"{root}/{self.name}/raw"
        if not Path(f"{path}/done").exists():
            self.download(path)
            Path(f"{path}/done").touch()

        # process, standardize and split
        path = f"{root}/{name}/processed/db"
        if not Path(f"{path}/done").exists():
            db = self.standardize_db(self.process_db())

            # save database splits independently
            db_splits = self.split_db(db)
            for split, db in db_splits.items():
                db.save(f"{path}/{split}")
            Path(f"{path}/done").touch()

        # load database splits
        self.db_splits = {
            split: rtb.data.database.Database.load(f"{path}/{split}")
            for split in ["train", "val", "test"]
        }

        # create tasks for each split
        self.task_splits = defaultdict(dict)
        for task_name in task_names:
            for split in ["train", "val", "test"]:
                path = f"{root}/{name}/processed/tasks/{task_name}/{split}"

                # save task split independent of other splits and tasks
                if not Path(f"{path}/done").exists():
                    task = self.task_fns[task_name](self.db_splits[split])
                    task.save(path)
                    Path(f"{path}/done").touch()

                # load task split
                self.task_splits[task_name][split] = Task.load(path)

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
        1. Add ctime column based on first temporal interaction if not present.
        2. Sort by ctime.
        3. Add primary key column if not present.
        4. Re-index primary key column with 0-indexed ints, if required.
        """

        raise NotImplementedError

    def split_db(
        self, db: rtb.data.database.Database
    ) -> dict[str, rtb.data.database.Database]:
        r"""Splits the database into train, val, and test splits."""

        assert sum(self.splits.values()) == 1.0

        # get time stamps for splits
        self.val_split_time = db.time_of_split(splits["train"])
        self.test_split_time = db.time_of_split(splits["train"] + splits["val"])

        # split the database
        db_train, db_val_test = db.split_at(self.val_split_time)
        db_val, db_test = val_test.split_at(self.test_split_time)

        return {"train": db_train, "val": db_val, "test": db_test}

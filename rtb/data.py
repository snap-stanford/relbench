from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

import pandas as pd


class SemanticType(Enum):
    r"""The semantic type of a database column."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME = "time"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"


class TaskType(Enum):
    r"""The type of a task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class Table:
    r"""A table in a database."""

    df: pd.DataFrame
    stypes: dict[str, SemanticType]  # column name -> semantic type
    fkeys: dict[str, str]  # column name -> table name
    ctime_col: "creation_time"  # name of column storing creation time

    def validate(self) -> bool
        r"""Validate the table.

        Check:
        1. Columns of df match keys of stypes.
        2. Foreign keys in stypes match keys of fkeys.
        3. There is exactly one primary key.
        4. ctime column exists and has stype of time.
        5. Columns are unique.
        """

        raise NotImplementedError

    @property
    def primary_key(self) -> str:
        r"""Return the name of the primary key."""

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the table to a parquet file. Stores stypes and fkeys as
        parquet metadata."""

        assert str(path).endswith(".parquet")
        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Table:
        r"""Loads a table from a parquet file."""

        assert str(path).endswith(".parquet")
        raise NotImplementedError

    def split_at(self, time_stamp: int) -> tuple[Table, Table]:
        r"""Splits the table into past (ctime <= time_stamp) and
        future (ctime > time_stamp) tables."""

        raise NotImplementedError


@dataclass
class Database:
    r"""A database is a collection of named tables linked by foreign key -
    primary key connections."""

    tables: dict[str, Table]

    def validate(self) -> bool:
        r"""Validate the database.

        Check:
        1. All tables validate.
        2. All foreign keys point to tables that exist.
        """

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the database to a directory. Simply saves each table
        individually with the table name as base name of file."""

        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Database:
        r"""Loads a database from a directory of tables in parquet files."""

        raise NotImplementedError

    def __add__(self, other: Database) -> Database:
        r"""Combines two databases with the same schema be concatenating rows of
        matching tables.

        The input Database objects are not modified."""

        raise NotImplementedError

    def time_of_split(self, frac: float) -> int:
        r"""Returns the time stamp before which there are (roughly) frac
        fractions of rows in the database."""

        raise NotImplementedError

    def split_at(self, time_stamp: int) -> tuple[Database, Database]:
        r"""Splits the database into past (ctime <= time_stamp) and
        future (ctime > time_stamp) databases."""

        raise NotImplementedError


@dataclass
class Task:
    r"""A task on a database. Can represent tasks on single (v1) or multiple
    (v2) entities."""

    task_type: TaskType
    metric: str
    time_stamps: list[int]  # time stamps to evaluate on
    entities: dict[str, list[str]]  # table name -> list of primary key values
    labels: list[int | float]  # list of labels
    num_classes: int = 0  # number of classes for classification tasks

    def validate(self) -> bool:
        r"""Validate the task.

        Check:
        1. Lengths of time_stamps, entities[<name>], and labels match.
        2. num_classes != 0 for classification tasks.
        """

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the task as a json file."""

        assert str(path).endswith(".json")
        raise NotImplementedError

    @staticmethod
    def load(self, path: str | os.PathLike) -> Task:
        r"""Loads a task from a json file."""

        assert str(path).endswith(".json")
        raise NotImplementedError


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
    task_fns: dict[str, callable[[Database], Task]]

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
            split: Database.load(f"{path}/{split}")
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

    def process_db(self) -> Database:
        r"""Processes the raw data into a database. To be implemented by
        subclass."""

        raise NotImplementedError

    def standardize_db(self, db: Database) -> Database:
        r"""
        1. Add ctime column based on first temporal interaction if not present.
        2. Sort by ctime.
        3. Add primary key column if not present.
        4. Re-index primary key column with 0-indexed ints, if required.
        """

        raise NotImplementedError

    def split_db(self, db: Database) -> dict[str, Database]:
        r"""Splits the database into train, val, and test splits."""

        assert sum(self.splits.values()) == 1.0

        # get time stamps for splits
        self.val_split_time = db.time_of_split(splits["train"])
        self.test_split_time = db.time_of_split(splits["train"] + splits["val"])

        # split the database
        db_train, db_val_test = db.split_at(self.val_split_time)
        db_val, db_test = val_test.split_at(self.test_split_time)

        return {"train": db_train, "val": db_val, "test": db_test}


import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .database import Database


class Dataset:
    r"""A dataset is a database with validation and test timestamps defined for it.

    Attributes:
        val_timestamp: Rows upto this timestamp (inclusive) can be input for validation.
        test_timestamp: Rows upto this timestamp (inclusive) can be input for testing.

    Validation split of a task involves predicting the target variable for a
    time period after val_timestamp (exclusive) using data upto val_timestamp.
    Similarly for test_timestamp.
    """

    # To be set by subclass.
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ) -> None:
        r"""Create a dataset object.

        Args:
            cache_dir: A directory for caching the database object. If specified,
                we will either process and cache the file (if not available) or use
                the cached file. If None, we will not use cached file and re-process
                everything from scratch without saving the cache.
        """

        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def validate_and_correct_db(self, db):
        r"""Validate and correct input db in-place.

        Removing rows after test_timestamp can result in dangling foreign keys.
        """
        # Validate that all primary keys are consecutively index.

        for table_name, table in db.table_dict.items():
            if table.pkey_col is not None:
                ser = table.df[table.pkey_col]
                if not (ser.values == np.arange(len(ser))).all():
                    raise RuntimeError(
                        f"The primary key column {table.pkey_col} of table "
                        f"{table_name} is not consecutively index."
                    )

        # Discard any foreign keys that are larger than primary key table as
        # dangling foreign keys (represented as None).
        for table_name, table in db.table_dict.items():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                num_pkeys = len(db.table_dict[pkey_table_name])
                mask = table.df[fkey_col] >= num_pkeys
                if mask.any():
                    table.df.loc[mask, fkey_col] = None

    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp=True) -> Database:
        r"""Return the database object.

        The returned database object is cached in memory.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: The database object.

        `upto_test_timestamp` is True by default to prevent test leakage.
        """

        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            print(f"Loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

        else:
            print("Making Database object from scratch...")
            print(
                "(You can also use `get_dataset(..., download=True)` "
                "for datasets prepared by the RelBench team.)"
            )
            tic = time.time()
            db = self.make_db()
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                print(f"Caching Database object to {db_path}...")
                tic = time.time()
                db.save(db_path)
                toc = time.time()
                print(f"Done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        self.validate_and_correct_db(db)

        return db

    def make_db(self) -> Database:
        r"""Make the database object from scratch, i.e. using raw data sources.

        To be implemented by subclass.
        """
        raise NotImplementedError

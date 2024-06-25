from functools import cached_property
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pooch

from relbench.data.database import Database


class Dataset:
    def __init__(
        self,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        cache_dir: str | os.PathLike | None = None,
    ) -> None:
        r"""Class holding database and task table construction logic.

        Args:
            val_timestamp (pd.Timestamp): The first timestamp for making val table.
            test_timestamp (pd.Timestamp): The first timestamp for making test table.
            cache_dir (str, optional): The directory to cache the dataset.
        """
        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp
        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + f"  val_timestamp={self.val_timestamp},\n"
            + f"  test_timestamp={self.test_timestamp},\n"
            + f"  cache_dir={self.cache_dir},\n"
            + ")"
        )

    def _make_db(self) -> Database:
        raise NotImplementedError

    @cached_property
    def db(self) -> Database:
        if self.cache_dir is None:
            db = self._make_db()
            db.reindex_pkeys_and_fkeys()
        else:
            db_path = f"{self.cache_dir}/db"
            if not Path(db_path).exists():
                db = self._make_db()
                db.reindex_pkeys_and_fkeys()
                db.save(db_path)
            else:
                db = Database.load(db_path)

        self._full_db = db
        db = db.upto(self.test_timestamp)

        return db

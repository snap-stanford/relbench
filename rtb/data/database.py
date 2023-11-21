import os
import time
from functools import cache
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from typing_extensions import Self

from rtb.data.table import Table


class Database:
    r"""A database is a collection of named tables linked by foreign key -
    primary key connections."""

    def __init__(self, table_dict: Dict[str, Table]) -> None:
        r"""Creates a database from a dictionary of tables."""

        self.table_dict = table_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Saves the database to a directory. Simply saves each table
        individually with the table name as base name of file."""

        for name, table in self.table_dict.items():
            print(f"saving table {name}...")
            tic = time.time()
            table.save(f"{path}/{name}.parquet")
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Loads a database from a directory of tables in parquet files."""

        table_dict = {}
        for table_path in Path(path).glob("*.parquet"):
            print(f"loading table {table_path}...")
            tic = time.time()
            table = Table.load(table_path)
            table_dict[table_path.stem] = table
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        return cls(table_dict)

    def time_cutoff(self, time: int) -> Self:
        r"""Returns a database with all rows upto time."""

        return Database(
            {name: table.time_cutoff(time) for name, table in self.table_dict.items()}
        )

    @property
    @cache
    def min_time(self) -> pd.Timestamp:
        r"""Returns the earliest timestamp in the database."""

        return min(table.min_time for table in self.table_dict.values())

    @property
    @cache
    def max_time(self) -> pd.Timestamp:
        r"""Returns the latest timestamp in the database."""

        return max(table.max_time for table in self.table_dict.values())

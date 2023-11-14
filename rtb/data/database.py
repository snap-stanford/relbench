import os
import time
from pathlib import Path
from typing import Dict, Tuple, Union

from rtb.data.table import Table
from typing_extensions import Self


class Database:
    r"""A database is a collection of named tables linked by foreign key -
    primary key connections."""

    def __init__(self, tables: Dict[str, Table]) -> None:
        r"""Creates a database from a dictionary of tables."""

        self.tables = tables

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def validate(self) -> bool:
        r"""Validate the database.

        Check:
        1. All tables validate.
        2. All foreign keys point to tables that exist.
        """

        raise NotImplementedError

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Saves the database to a directory. Simply saves each table
        individually with the table name as base name of file."""

        for name, table in self.tables.items():
            print(f"saving table {name}...")
            tic = time.time()
            table.save(f"{path}/{name}.parquet")
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Loads a database from a directory of tables in parquet files."""

        tables = {}
        for table_path in Path(path).glob("*.parquet"):
            print(f"loading table {table_path}...")
            tic = time.time()
            table = Table.load(table_path)
            tables[table_path.stem] = table
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        return cls(tables)

    def time_cutoff(self, time: int) -> Self:
        r"""Returns a database with all rows upto time."""

        return Database(
            {name: table.time_cutoff(time) for name, table in self.tables.items()}
        )

    def get_time_range(self) -> Tuple[int, int]:
        r"""Returns the earliest and latest timestamp in the database."""

        global_min_time = None
        global_max_time = None

        for table in self.tables.values():
            if table.time_col is None:
                continue

            min_time, max_time = table.get_time_range()

            if global_min_time is None or min_time < global_min_time:
                global_min_time = min_time

            if global_max_time is None or max_time > global_max_time:
                global_max_time = max_time

        return global_min_time, global_max_time

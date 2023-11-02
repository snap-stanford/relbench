from dataclasses import dataclass
import os

import rtb


@dataclass
class Database:
    r"""A database is a collection of named tables linked by foreign key -
    primary key connections."""

    tables: dict[str, rtb.data.table.Table]

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

    def time_cutoff(self, time: int) -> Database:
        r"""Returns a database with all rows upto time."""

        return {name: table.time_cutoff(time) for name, table in self.tables.items()}

    def get_time_range(self) -> tuple[int, int]:
        r"""Returns the earliest and latest timestamp in the database."""

        global_min_time = float("inf")
        global_max_time = 0

        for table in self.tables.values():
            if table.time_col is None:
                continue
            min_time, max_time = table.get_time_range()
            global_min_time = min(global_min_time, min_time)
            global_max_time = max(global_max_time, max_time)

        return global_min_time, global_max_time

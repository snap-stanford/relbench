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

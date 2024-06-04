import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from typing_extensions import Self

from relbench.data.table import Table


class Database:
    r"""A database is a collection of named tables linked by foreign key -
    primary key connections."""

    # TODO: maybe add a function to visualize schema in jupyter

    def __init__(self, table_dict: Dict[str, Table]) -> None:
        r"""Creates a database from a dictionary of tables."""

        self.table_dict = table_dict

    def __repr__(self) -> str:
        # TODO: add more info
        return f"{self.__class__.__name__}()"

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Saves the database to a directory. Simply saves each table
        individually with the table name as base name of file."""

        for name, table in self.table_dict.items():
            table.save(f"{path}/{name}.parquet")

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Loads a database from a directory of tables in parquet files."""

        table_dict = {}
        for table_path in Path(path).glob("*.parquet"):
            table = Table.load(table_path)
            table_dict[table_path.stem] = table

        return cls(table_dict)

    @property
    @lru_cache(maxsize=None)
    def min_timestamp(self) -> pd.Timestamp:
        r"""Returns the earliest timestamp in the database."""

        return min(
            table.min_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    def max_timestamp(self) -> pd.Timestamp:
        r"""Returns the latest timestamp in the database."""

        return max(
            table.max_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    @property
    @lru_cache(maxsize=None)
    def max_timestamp(self) -> pd.Timestamp:
        r"""Returns the latest timestamp in the database."""

        return max(
            table.max_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    def upto(self, time_stamp: pd.Timestamp) -> Self:
        r"""Returns a database with all rows upto time_stamp."""

        return Database(
            table_dict={
                name: table.upto(time_stamp) for name, table in self.table_dict.items()
            }
        )

    def reindex_pkeys_and_fkeys(self) -> None:
        r"""Mapping primary and foreign keys into indices according to
        the ordering in the primary key tables.
        """
        # Get pkey to idx mapping:
        index_map_dict: Dict[str, pd.Series] = {}
        for table_name, table in self.table_dict.items():
            if table.pkey_col is not None:
                if table.time_col is not None:
                    table.df = table.df.sort_values(table.time_col).reset_index(
                        drop=True
                    )

                ser = table.df[table.pkey_col]

                if ser.nunique() != len(ser):
                    raise RuntimeError(
                        f"The primary key '{table.pkey_col}' "
                        f"of table '{table_name}' contains "
                        "duplicated elements"
                    )
                arange_ser = pd.RangeIndex(len(ser)).astype("Int64")
                index_map_dict[table_name] = pd.Series(
                    index=ser,
                    data=arange_ser,
                    name="index",
                )
                table.df[table.pkey_col] = arange_ser

        # Replace fkey_col_to_pkey_table with indices.
        for table in self.table_dict.values():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                out = pd.merge(
                    table.df[fkey_col],
                    index_map_dict[pkey_table_name],
                    how="left",
                    left_on=fkey_col,
                    right_index=True,
                )
                table.df[fkey_col] = out["index"]

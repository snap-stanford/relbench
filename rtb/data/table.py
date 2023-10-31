from dataclasses import dataclass
from enum import Enum
import os

import pandas as pd


class SemanticType(Enum):
    r"""The semantic type of a database column."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MULTI_CATEGORICAL = "multi_categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME = "time"


@dataclass
class Table:
    r"""A table in a database."""

    df: pd.DataFrame
    pkey: str  # name of primary key column
    fkeys: dict[str, str]  # column name -> table name
    feat_cols: dict[str, SemanticType]  # column name -> semantic type
    time_col: str | None = None  # name of column storing creation time

    def validate(self) -> bool:
        r"""Validate the table."""

        raise NotImplementedError

    def save(self, path: str | os.PathLike) -> None:
        r"""Saves the table to a parquet file. Stores other attributes as
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

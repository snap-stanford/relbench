from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, Union, Optional

import pandas as pd


class SemanticType(Enum):
    r"""The semantic type of a database column."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MULTI_CATEGORICAL = "multi_categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME = "time"

import os
import pandas as pd

class Table:
    r"""A table in a database."""

    def __init__(self, 
                 df: pd.DataFrame, 
                 feat_cols: Dict[str, str], 
                 fkeys: Dict[str, str], 
                 pkey: Optional[str], 
                 time_col: Optional[str] = None):
        self.df = df
        self.feat_cols = feat_cols
        self.fkeys = fkeys
        self.pkey = pkey
        self.time_col = time_col

    def __repr__(self):
        return (f"Table(df={self.df}, feat_cols={self.feat_cols}, fkeys={self.fkeys}, "
                f"pkey={self.pkey}, time_col={self.time_col})")

    def validate(self) -> bool:
        r"""Validate the table."""
        # Check if pkey exists
        if self.pkey and self.pkey not in self.df.columns:
            return False
        # Check if feat_cols exist
        for col in self.feat_cols:
            if col not in self.df.columns:
                return False
        # Check if fkeys columns exist
        for col in self.fkeys:
            if col not in self.df.columns:
                return False
        return True

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Saves the table to a parquet file. Stores other attributes as
        parquet metadata."""
        assert str(path).endswith(".parquet")
        metadata = {
            'feat_cols': self.feat_cols,
            'fkeys': self.fkeys,
            'pkey': self.pkey,
            'time_col': self.time_col
        }
        self.df.to_parquet(path, index=False, metadata=metadata)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> 'Table':
        r"""Loads a table from a parquet file."""
        assert str(path).endswith(".parquet")
        df = pd.read_parquet(path)
        metadata = df.columns.metadata
        return Table(df, metadata['feat_cols'], metadata['fkeys'], metadata['pkey'], metadata['time_col'])

    def split_at(self, time_stamp: int) -> tuple['Table', 'Table']:
        r"""Splits the table into past (ctime <= time_stamp) and
        future (ctime > time_stamp) tables."""
        if not self.time_col:
            raise ValueError("No time column specified for splitting.")
        past_df = self.df[self.df[self.time_col] <= time_stamp]
        future_df = self.df[self.df[self.time_col] > time_stamp]
        past_table = Table(past_df, self.feat_cols, self.fkeys, self.pkey, self.time_col)
        future_table = Table(future_df, self.feat_cols, self.fkeys, self.pkey, self.time_col)
        return past_table, future_table

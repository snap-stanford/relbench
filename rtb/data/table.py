from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, Union, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

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
        
        # Check if time_col is a timestamp column, if not convert it
        if self.time_col:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_col]):
                self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

    def __repr__(self):
        return (f"Table(df={self.df.head()}, feat_cols={self.feat_cols}, fkeys={self.fkeys}, "
                f"pkey={self.pkey}, time_col={self.time_col})")
    
    def __len__(self) -> int:
        """Returns the number of rows in the table (DataFrame)."""
        return len(self.df)

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
        """Saves the table to a parquet file. Stores other attributes as
        parquet metadata."""
        assert str(path).endswith(".parquet")
        metadata = {
            'feat_cols': self.feat_cols,
            'fkeys': self.fkeys,
            'pkey': self.pkey,
            'time_col': self.time_col
        }

        # Convert DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(self.df)

        # Add metadata to the PyArrow Table
        metadata_bytes = {key: json.dumps(value).encode('utf-8') for key, value in metadata.items()}
        table = table.replace_schema_metadata(metadata_bytes)

        # Write the PyArrow Table to a Parquet file using pyarrow.parquet
        pq.write_table(table, path)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> 'Table':
        """Loads a table from a parquet file."""
        assert str(path).endswith(".parquet")

        # Read the Parquet file using pyarrow
        table = pa.parquet.read_table(path)
        df = table.to_pandas()

        # Extract metadata
        metadata_bytes = table.schema.metadata
        metadata = {key.decode('utf-8'): json.loads(value.decode('utf-8'))
                for key, value in metadata_bytes.items()}
        return Table(df, metadata['feat_cols'], metadata['fkeys'], metadata['pkey'], metadata['time_col'])

    def split_at(self, time_stamp: Union[int, str, pd.Timestamp]) -> Tuple['Table', 'Table']:
        """Splits the table into past (ctime <= time_stamp) and
        future (ctime > time_stamp) tables."""
        if not self.time_col:
            raise ValueError("No time column specified for splitting.")

        # Convert time_stamp to pd.Timestamp if it's not already
        time_stamp = pd.Timestamp(time_stamp)

        past_df = self.df[self.df[self.time_col] <= time_stamp]
        future_df = self.df[self.df[self.time_col] > time_stamp]

        past_table = Table(past_df, self.feat_cols, self.fkeys, self.pkey, self.time_col)
        future_table = Table(future_df, self.feat_cols, self.fkeys, self.pkey, self.time_col)

        return past_table, future_table

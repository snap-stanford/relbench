import copy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
from typing_extensions import Self

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json


class Table:
    r"""A table in a database."""

    def __init__(
        self,
        df: pd.DataFrame,
        fkeys: Dict[str, str],
        pkey: Union[str, None],
        time_col: Union[str, None] = None,
    ):
        self.df = df
        self.fkeys = fkeys
        self.pkey = pkey
        self.time_col = time_col

    def __repr__(self):
        return f"Table(df=\n{self.df},\nfkeys={self.fkeys},\npkey={self.pkey},\ntime_col={self.time_col})"

    def __len__(self) -> int:
        """Returns the number of rows in the table (DataFrame)."""
        return len(self.df)

    def validate(self) -> bool:
        r"""Validate the table."""
        # Check if pkey exists
        if self.pkey and self.pkey not in self.df.columns:
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
            "fkeys": self.fkeys,
            "pkey": self.pkey,
            "time_col": self.time_col,
        }

        # Convert DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(self.df, preserve_index=False)

        # Add metadata to the PyArrow Table
        metadata_bytes = {
            key: json.dumps(value).encode("utf-8") for key, value in metadata.items()
        }

        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write the PyArrow Table to a Parquet file using pyarrow.parquet
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        """Loads a table from a parquet file."""
        assert str(path).endswith(".parquet")

        # Read the Parquet file using pyarrow
        table = pa.parquet.read_table(path)
        df = table.to_pandas()

        # Extract metadata
        metadata_bytes = table.schema.metadata
        metadata = {
            key.decode("utf-8"): json.loads(value.decode("utf-8"))
            for key, value in metadata_bytes.items()
            if key in [b"fkeys", b"pkey", b"time_col"]
        }
        return cls(
            df=df,
            fkeys=metadata["fkeys"],
            pkey=metadata["pkey"],
            time_col=metadata["time_col"],
        )

    def time_cutoff(self, time_stamp: int) -> Self:
        r"""Returns a table with all rows upto time."""

        if self.time_col is None:
            return self

        new_table = copy.copy(self)
        df = new_table.df
        df = df[df[self.time_col] <= time_stamp]
        new_table.df = df
        return new_table

    def get_time_range(self) -> Tuple[int, int]:
        r"""Returns the earliest and latest timestamp in the table."""

        assert self.time_col is not None

        ts = self.df[self.time_col]
        return ts.min(), ts.max()

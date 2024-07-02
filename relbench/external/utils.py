from typing import Any, Dict

import numpy as np
import pandas as pd
from torch_frame import stype
from torch_frame.utils import infer_df_stype

from relbench.data import Database, Table


def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = ser.astype(int).values
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def remove_pkey_fkey(col_to_stype: Dict[str, Any], table: Table) -> dict:
    r"""Remove pkey, fkey columns since they will not be used as input feature."""
    if table.pkey_col is not None:
        if table.pkey_col in col_to_stype:
            col_to_stype.pop(table.pkey_col)
    for fkey in table.fkey_col_to_pkey_table.keys():
        if fkey in col_to_stype:
            col_to_stype.pop(fkey)


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, stype]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        df = df.sample(min(1_000, len(df)))
        inferred_col_to_stype = infer_df_stype(df)
        # Hack for now. This is relevant for rel-amazon.
        for col, stype_ in inferred_col_to_stype.items():
            if stype_.value == "embedding":
                inferred_col_to_stype[col] = stype.multicategorical
        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype

    return inferred_col_to_stype_dict

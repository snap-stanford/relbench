import numpy as np
import pandas as pd


def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    assert str(ser.dtype).startswith("datetime64[s") or str(ser.dtype).startswith(
        "datetime64[ns"
    )
    unix_time = ser.astype(int).values
    if str(ser.dtype).startswith("datetime64[ns"):
        unix_time //= 10**9
    return unix_time
import shutil
from pathlib import Path
from typing import Union

import pooch


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    shutil.unpack_archive(zip_path, unzip_path)
    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
    ]

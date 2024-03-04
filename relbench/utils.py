import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pandas as pd
import pooch


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    if action != "fetch":
        shutil.unpack_archive(zip_path, unzip_path)
    else:  # fetch
        try:  # sanity check if all files are fully extracted comparing size
            for f in ZipFile(zip_path).infolist():
                if not f.is_dir():
                    fsize = os.path.getsize(os.path.join(unzip_path, f.filename))
                    assert f.file_size == fsize
        except Exception:  # otherwise do full unpack
            shutil.unpack_archive(zip_path, unzip_path)

    return unzip_path


def unzip_and_convert_csv_to_parquet_processor(
    fname: Union[str, Path], action: str, pooch: pooch.Pooch
) -> Path:
    unzip_path = unzip_processor(fname, action, pooch)

    # Convert csv to parquet
    for csv_file in unzip_path.glob("**/*.csv"):
        parquet_file = csv_file.with_suffix(".parquet")
        if not parquet_file.exists():  # Only convert if parquet file does not exist
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file)

    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
    ]

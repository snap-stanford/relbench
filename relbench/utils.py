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


def clean_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    r"""Clean the time column of a pandas dataframe.
    Args:
        df (pd.DataFrame): The pandas dataframe to clean the timecolumn for.
        col (str): The time column name.

    Returns:
        (pd.DataFrame): The pandas dataframe with the cleaned time column.
    """
    # Change time column to pd timestamp series
    # Attempt to convert "CreationDate" to datetime format
    df[col] = pd.to_datetime(df[col], errors="coerce")

    # Count the number of comments before removing invalid dates
    total_before = len(df)

    # Remove rows where "CreationDate" is NaT (indicating parsing failure)
    df = df.dropna(subset=[col])

    # Count the number of comments after removing invalid dates
    total_after = len(df)

    # Calculate the percentage of comments removed
    percentage_removed = ((total_before - total_after) / total_before) * 100

    # Print the percentage of comments removed
    print(
        f"Percentage of rows removed due to invalid dates: "
        f"{percentage_removed:.2f}%"
    )
    return df

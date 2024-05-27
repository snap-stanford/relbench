import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pooch


def decompress_gz_file(input_path: str, output_path: str):
    import gzip
    import shutil

    # Open the gz file in binary read mode
    with gzip.open(input_path, "rb") as f_in:
        # Open the output file in binary write mode
        with open(output_path, "wb") as f_out:
            # Copy the decompressed data from the gz file to the output file
            shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed file saved as: {output_path}")


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


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
    ]

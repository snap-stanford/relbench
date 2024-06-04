import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pooch

from relbench.data import Database


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


def clean_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    r"""Clean the time column of a pandas dataframe.
    Args:
        df (pd.DataFrame): The pandas dataframe to clean the timecolumn for.
        col (str): The time column name.

    Returns:
        (pd.DataFrame): The pandas dataframe with the cleaned time column.
    """
    df[col] = pd.to_datetime(df[col], errors="coerce")

    # Count the number of rows before removing invalid dates
    total_before = len(df)

    # Remove rows where timestamp is NaT (indicating parsing failure)
    df = df.dropna(subset=[col])

    # Count the number of rows after removing invalid dates
    total_after = len(df)

    # Calculate the percentage of rows removed
    percentage_removed = ((total_before - total_after) / total_before) * 100

    # Print the percentage of comments removed
    print(
        f"Percentage of rows removed due to invalid dates: "
        f"{percentage_removed:.2f}%"
    )
    return df


def visualize_database_schema(db: Database, path: str):
    r"""Visualize a database schema and save the figure to path."""
    G = nx.Graph()
    for table_name in db.table_dict:
        G.add_node(table_name, name=table_name)
    breakpoint()
    for table_name, table in db.table_dict.items():
        pkey_table_dict = table.fkey_col_to_pkey_table
        for _, pkey_table_name in pkey_table_dict.items():
            G.add_edge(table_name, pkey_table_name)
    nx.draw(G, labels=nx.get_node_attributes(G, "name"))
    plt.savefig(path)

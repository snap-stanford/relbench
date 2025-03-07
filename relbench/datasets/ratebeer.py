import os
from typing import Dict, List, Optional, Union
import zipfile

import numpy as np
import pandas as pd
import pooch
from tqdm import tqdm

from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor

# Update URL to Dropbox direct download link
DB_URL = "https://www.dropbox.com/scl/fi/exwygxep7vdvq55uiq28r/db.zip?rlkey=o7q0r8nw758p4wxx1wka9ubuj&st=rg3gvkxg&dl=1"
CACHE_DIR = "/lfs/madmax2/0/akhatua/relbench_cache/rel-ratebeer/testing"

class RateBeerDataset(Dataset):
    # Dataset spans approximately 2000-2024 based on analysis
    # Using 2018 and 2020 as sensible train/val/test splits (80%/10%/10% approx)
    val_timestamp = pd.Timestamp("2018-01-01")
    test_timestamp = pd.Timestamp("2020-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Set up pooch to use our cache directory
        path = pooch.retrieve(
            DB_URL,
            known_hash="c3921164da60f8c97e6530d1f2872f7e0d307f8276348106db95c10c2df677ad",
            progressbar=True,
            processor=unzip_processor,
            path=CACHE_DIR
        )
        
        print("Reading from processed database...")
        tables = {}
        
        # Define table configurations
        table_configs = {
            # Reference tables
            "countries": {"pkey": "country_id", "fkeys": {}, "time_col": None},
            "states": {"pkey": "state_id", "fkeys": {"country_id": "countries"}, "time_col": None},
            "beer_styles": {"pkey": "style_id", "fkeys": {"parent_style_id": "beer_styles"}, "time_col": None},
            "glassware": {"pkey": "glassware_id", "fkeys": {}, "time_col": None},
            "place_types": {"pkey": "type_id", "fkeys": {}, "time_col": None},
            
            # Core tables
            "places": {
                "pkey": "place_id",
                "fkeys": {
                    "country_id": "countries",
                    "state_id": "states",
                    "type_id": "place_types"
                },
                "time_col": "created_at"
            },
            "beers": {
                "pkey": "beer_id",
                "fkeys": {
                    "style_id": "beer_styles",
                    "brewer_id": "brewers",
                    "contract_brewer_id": "brewers"
                },
                "time_col": "created_at"
            },
            "beer_aliases": {
                "pkey": ["root_beer_id", "alias_beer_id"],
                "fkeys": {
                    "root_beer_id": "beers",
                    "alias_beer_id": "beers"
                },
                "time_col": None
            },
            "beer_ratings": {
                "pkey": "rating_id",
                "fkeys": {
                    "beer_id": "beers",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "beer_upcs": {
                "pkey": "upc",
                "fkeys": {"beer_id": "beers"},
                "time_col": None
            },
            "availability": {
                "pkey": "avail_id",
                "fkeys": {
                    "beer_id": "beers",
                    "country_id": "countries",
                    "place_id": "places"
                },
                "time_col": "created_at"
            },
            "favorites": {
                "pkey": "favorite_id",
                "fkeys": {
                    "beer_id": "beers",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "place_ratings": {
                "pkey": "rating_id",
                "fkeys": {
                    "place_id": "places",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "brewers": {
                "pkey": "brewer_id",
                "fkeys": {"state_id": "states", "country_id": "countries"},
                "time_col": "created_at"
            },
            # Users table combines information about users who rate beers and places
            "users": {
                "pkey": "user_id",
                "fkeys": {},
                "time_col": "created_at"
            }
        }

        # Read tables from extracted directory
        for table_name, config in tqdm(table_configs.items(), desc="Loading tables", total=len(table_configs)):
            csv_path = os.path.join(path, "db", f"{table_name}.csv")
            df = pd.read_csv(csv_path, low_memory=False)
                
            # Convert timestamp columns if present
            if config["time_col"] is not None:
                for col in ["created_at", "updated_at", "last_edited_at", "opened_at"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format='mixed')
                
            tables[table_name] = Table(
                df=df,
                fkey_col_to_pkey_table=config["fkeys"],
                pkey_col=config["pkey"],
                time_col=config["time_col"]
            )
            tqdm.write(f"Loaded {table_name}: {len(df):,} rows")

        print("\nAll tables loaded successfully!")
        return Database(tables)


if __name__ == "__main__":
    ratebeer_dataset = RateBeerDataset()
    ratebeer_db = ratebeer_dataset.make_db()

    # Print table statistics
    for table_name, table in ratebeer_db.table_dict.items():
        print(f'Number of rows in {table_name} table: {table.df.shape[0]}')
        print(f'Sample of {table_name} table:')
        print(table.df.head())
        print()

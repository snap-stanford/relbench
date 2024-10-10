import os
import shutil
from pathlib import Path
from functools import lru_cache

import pandas as pd

from relbench.base import Database, Dataset, Table


class HMDataset(Dataset):
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )
    # Train for the most recent 1 year out of 2 years of the original
    # time period
    val_timestamp = pd.Timestamp("2020-09-07")
    test_timestamp = pd.Timestamp("2020-09-14")

    def make_db(self) -> Database:
        path = os.path.join("data", "hm-recommendation")
        zip = os.path.join(path, "h-and-m-personalized-fashion-recommendations.zip")
        customers = os.path.join(path, "customers.csv")
        articles = os.path.join(path, "articles.csv")
        transactions = os.path.join(path, "transactions_train.csv")
        if not os.path.exists(customers):
            if not os.path.exists(zip):
                raise RuntimeError(
                    f"Dataset not found. Please download "
                    f"h-and-m-personalized-fashion-recommendations.zip from "
                    f"'{self.url}' and move it to '{path}'. Once you have your"
                    f"Kaggle API key, you can use the following command: "
                    f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations"
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)

        articles_df = pd.read_csv(articles)
        customers_df = pd.read_csv(customers)
        transactions_df = pd.read_csv(transactions)
        transactions_df["t_dat"] = pd.to_datetime(
            transactions_df["t_dat"], format="%Y-%m-%d"
        )

        db = Database(
            table_dict={
                "article": Table(
                    df=articles_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="article_id",
                ),
                "customer": Table(
                    df=customers_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
                ),
                "transactions": Table(
                    df=transactions_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "article_id": "article",
                    },
                    time_col="t_dat",
                ),
            }
        )

        db = db.from_(pd.Timestamp("2019-09-07"))

        return db
    
class HMMinusColumnsDataset(HMDataset):
    r"""The H&M dataset without the price and article_id columns in the transactions table."""

    @property
    def remove_columns_dict(self) -> dict:
        """Property to define columns to remove. Subclasses must implement this."""
        raise NotImplementedError
    
    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp=True) -> Database:
        db = super().get_db(upto_test_timestamp)
        
        for table_name, columns in self.remove_columns_dict.items():
            # check if any of the columns to be removed are not in the table
            if not all(col in db.table_dict[table_name].df.columns for col in columns):
                continue

            # save the columns to be dropped
            table_primary_key = db.table_dict[table_name].pkey_col
            if not table_primary_key:
                raise ValueError(f"Primary key not found for table {table_name}.")
            db.table_dict[table_name].removed_cols = db.table_dict[table_name].df[[table_primary_key] + columns]

            # drop the columns
            # db.table_dict[table_name].df = db.table_dict[table_name].df.drop(columns=columns)

            # # remove 'fkey_col_to_pkey_table' entries for the removed columns
            # for col in columns:
            #     db.table_dict[table_name].fkey_col_to_pkey_table.pop(col, None)

        return db
    
class HMMinusPriceDataset(HMMinusColumnsDataset):
    r"""The H&M dataset without the price column in the transactions table."""

    @property
    def remove_columns_dict(self) -> dict:
        return {"transactions": ["price"]}
    
    def make_db(self) -> Database:
        db = super().make_db()
        
        # add transaction_id as primary key to transactions
        db.table_dict["transactions"].df["transaction_id"] = range(len(db.table_dict["transactions"]))
        # make transaction_id be the first column
        db.table_dict["transactions"].df = db.table_dict["transactions"].df[["transaction_id"] + [col for col in db.table_dict["transactions"].df.columns if col != "transaction_id"]]
        db.table_dict["transactions"].pkey_col = "transaction_id"
        return db
    
class HMMinusArticleIDDataset(HMMinusColumnsDataset):
    r"""The H&M dataset without the article_id column in the transactions table."""

    @property
    def remove_columns_dict(self) -> dict:
        return {"transactions": ["article_id"]}

    def make_db(self) -> Database:
        db = super().make_db()

        # add transaction_id as primary key to transactions
        db.table_dict["transactions"].df["transaction_id"] = range(len(db.table_dict["transactions"]))
        # make transaction_id be the first column
        db.table_dict["transactions"].df = db.table_dict["transactions"].df[["transaction_id"] + [col for col in db.table_dict["transactions"].df.columns if col != "transaction_id"]]
        db.table_dict["transactions"].pkey_col = "transaction_id"
        return db
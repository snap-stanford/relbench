import copy
import json
import os
import re
import time

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.json

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset


class LTV(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user reviews in the time_frame."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=TaskType.REGRESSION,
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for LTV."""

        # XXX: If this is not fast enough, we can try using duckdb to query the
        # parquet files directly.

        # columns in time_window_df: time_offset, time_cutoff

        # XXX: can we directly access tables in the sql string?
        product = db.tables["product"].df
        review = db.tables["review"].df

        # due to query optimization and parallelization,
        # this should be fast enough
        # and doing sql queries is also flexible enough to easily implement
        # a variety of other tasks
        df = duckdb.sql(
            r"""
            SELECT
                time_offset,
                time_cutoff,
                customer_id,
                SUM(price) AS ltv
            FROM
                time_window_df,
                (
                    SELECT
                        review_time,
                        customer_id,
                        price
                    FROM
                        product,
                        review
                    WHERE
                        product.product_id = review.product_id
                ) AS tmp
            WHERE
                tmp.review_time > time_window_df.time_offset AND
                tmp.review_time <= time_window_df.time_cutoff
            GROUP BY customer_id, time_offset, time_cutoff
            """
        ).df()

        return Table(
            df=df,
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="time_offset",
        )


class ProductDataset(Dataset):
    name = "rtb-product"

    # raw file names
    product_file_name = "meta_Books.json"
    review_file_name = "Books.json"

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTV()}

    # TODO: implement get_cutoff_times()

    def download(self, path: str | os.PathLike) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raise NotImplementedError

    def process(self) -> Database:
        r"""Process the raw files into a database.

        Sample output to give an idea of the processing time:

        reading product info from data/rtb-product/raw/meta_Books.json...
        done in 1.12 seconds.
        converting to pandas dataframe...
        done in 5.88 seconds.
        processing product info...
        done in 1.44 seconds.
        reading review and reviewer info from data/rtb-product/raw/Books.json...
        done in 13.94 seconds.
        converting to pandas dataframe...
        done in 77.43 seconds.
        processing review and customer info...
        done in 41.19 seconds.

        # beyond this comes from another function

        saving table product...
        done in 6.15 seconds.
        saving table customer...
        done in 16.68 seconds.
        saving table review...
        done in 150.98 seconds.
        """

        tables = {}

        ### product table ###

        path = f"{self.root}/{self.name}/raw/{self.product_file_name}"
        print(f"reading product info from {path}...")
        tic = time.time()
        pa_table = pa.json.read_json(
            path,
            parse_options=pa.json.ParseOptions(
                explicit_schema=pa.schema(
                    [
                        ("asin", pa.string()),
                        ("category", pa.list_(pa.string())),
                        ("brand", pa.string()),
                        ("title", pa.string()),
                        ("description", pa.list_(pa.string())),
                        ("price", pa.string()),
                    ]
                ),
                unexpected_field_behavior="ignore",
            ),
        )
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"converting to pandas dataframe...")
        tic = time.time()
        df = pa_table.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"processing product info...")
        tic = time.time()

        # asin is not intuitive / recognizable
        df.rename(columns={"asin": "product_id"}, inplace=True)

        # remove products with missing price
        df = df.query("price != ''")

        # price is like "$x,xxx.xx"
        df.loc[:, "price"] = df["price"].apply(
            lambda x: float(x[1:].replace(",", "")) if x[0] == "$" else None
        )

        # remove Books from category because it's redundant
        # and empty list actually means missing category because atleast Books should be there
        df.loc[:, "category"] = df["category"].apply(
            lambda x: None if len(x) == 0 else [c for c in x if c != "Books"]
        )

        # description is either [] or ["some description"]
        df.loc[:, "description"] = df["description"].apply(
            lambda x: None if len(x) == 0 else x[0]
        )

        tables["product"] = Table(
            df=df,
            fkeys={},
            pkey="asin",
            time_col=None,
        )

        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        ### review table ###

        path = f"{self.root}/{self.name}/raw/{self.review_file_name}"
        print(f"reading review and reviewer info from {path}...")
        tic = time.time()
        pa_table = pa.json.read_json(
            path,
            parse_options=pa.json.ParseOptions(
                explicit_schema=pa.schema(
                    [
                        ("unixReviewTime", pa.int32()),
                        ("reviewerID", pa.string()),
                        ("reviewerName", pa.string()),
                        ("asin", pa.string()),
                        ("overall", pa.float32()),
                        ("verified", pa.bool_()),
                        ("reviewText", pa.string()),
                        ("summary", pa.string()),
                    ]
                ),
                unexpected_field_behavior="ignore",
            ),
        )
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"converting to pandas dataframe...")
        tic = time.time()
        df = pa_table.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"processing review and customer info...")
        tic = time.time()

        df.rename(
            columns={
                "unixReviewTime": "review_time",
                "reviewerID": "customer_id",
                "reviewerName": "customer_name",
                "asin": "product_id",
                "overall": "rating",
                "reviewText": "review_text",
            },
            inplace=True,
        )

        df.loc[:, "review_time"] = pd.to_datetime(df["review_time"], unit="s")

        # XXX: can we speed this up?
        cdf = df.loc[
            df.duplicated(subset=["customer_id"]), ["customer_id", "customer_name"]
        ]
        tables["customer"] = Table(
            df=cdf,
            fkeys={},
            pkey="customer_id",
            time_col=None,
        )

        df.drop(columns=["customer_name"], inplace=True)
        tables["review"] = Table(
            df=df,
            fkeys={
                "customer_id": "customer",
                "product_id": "product",
            },
            pkey=None,
            time_col="review_time",
        )

        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        return Database(tables)

import copy
import json
import os
import re
import time

from typing import Dict, Union
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.json

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset


class ChurnTask(Task):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    def __init__(self):
        super().__init__(
            target_col="churn",
            task_type=TaskType.BINARY_CLASSIFICATION,
            test_time_window_sizes=[pd.Timedelta("1W")],
            metrics=["auprc"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                customer_id,
                NOT EXISTS (review_time)
            FROM
                time_window_df,
                (
                    SELECT
                        review_time,
                        customer_id,
                    FROM
                        customer LEFT JOIN review ON customer_id
                ) AS tmp
            WHERE
                tmp.review_time > time_window_df.window_min_time AND
                tmp.review_time <= time_window_df.window_max_time
            GROUP BY customer_id, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="window_min_time",
        )


class LTVTask(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=TaskType.REGRESSION,
            test_time_window_sizes=[pd.Timedelta("1W")],
            metrics=["auprc"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
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
                tmp.review_time > time_window_df.window_min_time AND
                tmp.review_time <= time_window_df.window_max_time
            GROUP BY customer_id, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="window_min_time",
        )


class LTVTask(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user reviews in the time window."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=TaskType.REGRESSION,
            test_time_window_sizes=[pd.Timedelta("1W")],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for LTV."""

        # XXX: If this is not fast enough, we can try using duckdb to query the
        # parquet files directly.

        # columns in time_window_df: window_min_time, window_max_time

        # XXX: can we directly access tables in the sql string?
        product = db.tables["product"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
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
                tmp.review_time > time_window_df.window_min_time AND
                tmp.review_time <= time_window_df.window_max_time
            GROUP BY customer_id, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkeys={"customer_id": "customer"},
            pkey_col=None,
            time_col="window_min_time",
        )


class ProductDataset(Dataset):
    name = "rtb-product"

    def __init__(
        self,
        root: str | os.PathLike,
        process=False,
        product_file_name="meta_Books.json",
        review_file_name="Books_5.json",
    ):
        self.product_file_name = product_file_name
        self.review_file_name = review_file_name

        super().__init__(root, process)

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTVTask()}

    # TODO: implement get_cutoff_times()

    def download(self, path: Union[str, os.PathLike]) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raise NotImplementedError

    def process(self) -> Database:
        r"""Process the raw files into a database."""

        ### product table ###

        path = f"{self.root}/{self.name}/raw/{self.product_file_name}"
        print(f"reading product info from {path}...")
        tic = time.time()
        ptable = pa.json.read_json(
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
        pdf = ptable.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"processing product info...")
        tic = time.time()

        # asin is not intuitive / recognizable
        pdf.rename(columns={"asin": "product_id"}, inplace=True)

        # price is like "$x,xxx.xx", "$xx.xx", or "$xx.xx - $xx.xx", or garbage html
        # if it's a range, we take the first value
        pdf.loc[:, "price"] = pdf["price"].apply(
            lambda x: None
            if x is None or x == "" or x[0] != "$"
            else float(x.split(" ")[0][1:].replace(",", ""))
        )

        # remove products with missing price
        pdf = pdf.dropna(subset=["price"])

        pdf.loc[:, "category"] = pdf["category"].apply(
            lambda x: None if x is None or len(x) == 0 else x
        )

        # description is either [] or ["some description"]
        pdf.loc[:, "description"] = pdf["description"].apply(
            lambda x: None if x is None or len(x) == 0 else x[0]
        )

        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        ### review table ###

        path = f"{self.root}/{self.name}/raw/{self.review_file_name}"
        print(f"reading review and customer info from {path}...")
        tic = time.time()
        rtable = pa.json.read_json(
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
        rdf = rtable.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"processing review and customer info...")
        tic = time.time()

        rdf.rename(
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

        rdf.loc[:, "review_time"] = pd.to_datetime(rdf["review_time"], unit="s")

        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"keeping only products common to product and review tables...")
        tic = time.time()
        plist = list(set(pdf["product_id"]) & set(rdf["product_id"]))
        pdf.query("product_id in @plist", inplace=True)
        rdf.query("product_id in @plist", inplace=True)
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print(f"extracting customer table...")
        tic = time.time()
        cdf = (
            rdf[["customer_id", "customer_name"]]
            .drop_duplicates(subset=["customer_id"])
            .copy()
        )
        rdf.drop(columns=["customer_name"], inplace=True)
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        return Database(
            tables={
                "product": Table(
                    df=pdf,
                    fkeys={},
                    pkey_col="product_id",
                    time_col=None,
                ),
                "customer": Table(
                    df=cdf,
                    fkeys={},
                    pkey_col="customer_id",
                    time_col=None,
                ),
                "review": Table(
                    df=rdf,
                    fkeys={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                    pkey_col=None,
                    time_col="review_time",
                ),
            }
        )

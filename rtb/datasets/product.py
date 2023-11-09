import copy
import json
import os
import re
import time

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.json
from tqdm.auto import tqdm

tqdm.pandas()

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


def product_process(row):
    r"""Process a row in the product table."""
    ret = copy.copy(row)

    # convert description from list[str] to str
    if row["description"] is not None:
        try:
            # most are 0 or 1 length lists
            ret["description"] = row["description"][0]
        except IndexError:
            # empty list
            ret["description"] = None

    # parse price from string
    try:
        # if there's a price range, this simply parses the lower bound
        ret["price"] = float(row["price"].split(" ")[0][1:])
    except ValueError:
        # invalid price string
        ret["price"] = None

    return ret


def pa_read_json(path):
    return read_json(
        path,
        parse_options=ParseOptions(
            explicit_schema=pa.schema(
                [
                    ("category", pa.list_(pa.string())),
                    ("description", pa.list_(pa.string())),
                    ("asin", pa.string()),
                    ("brand", pa.string()),
                    ("title", pa.string()),
                    ("price", pa.string()),
                ]
            ),
            unexpected_field_behavior="ignore",
        ),
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
        r"""Process the raw files into a database."""

        tables = {}

        print(f"reading product info from {self.product_file_name}...")
        tic = time.time()
        pa_table = pa.json.read_json(
            f"{self.root}/{self.name}/raw/{self.product_file_name}",
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

        # remove products with missing price
        df = df.query("price != ''")

        df.progress_apply(product_process, axis=1)

        # some more prices may be detected as missing on failure to parse price string
        df.dropna(subset=["price"], inplace=True)

        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        asins = set(df["asin"])

        tables["product"] = Table(
            df=df,
            fkeys={},
            pkey="asin",
            time_col=None,
        )

        # TODO: refactor below

        # review table
        customers = {}
        reviews = []
        path = f"{self.root}/{self.name}/raw/{self.review_file_name}"
        with open(path, "r") as f:
            for l in tqdm(f, total=self.review_lines):
                raw = json.loads(l)

                # only keep reviews for products in product table
                if raw["asin"] not in product_ids:
                    continue

                try:
                    reviews.append(
                        {
                            "review_time": raw["unixReviewTime"],
                            "customer_id": raw["reviewerID"],
                            "product_id": raw["asin"],
                            "rating": raw["overall"],
                            "verified": raw["verified"],
                            "review_text": raw["reviewText"],
                            "summary": raw["summary"],
                        }
                    )
                    customers[raw["reviewerID"]] = raw["reviewerName"]
                except KeyError:
                    # ignoring missing data rows for now
                    pass

        tables["customer"] = Table(
            df=pd.DataFrame(
                {
                    "customer_id": list(customers.keys()),
                    "name": list(customers.values()),
                }
            ),
            fkeys={},
            pkey="customer_id",
            time_col=None,
        )

        tables["review"] = Table(
            df=pd.DataFrame(reviews),
            fkeys={
                "customer_id": "customer",
                "product_id": "product",
            },
            pkey=None,
            time_col="review_time",
        )

        return Database(tables)

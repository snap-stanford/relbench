import json
import os
import re

import duckdb
import pandas as pd
from tqdm.auto import tqdm

from rtb.data.table import SemanticType, Table
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
            feat_cols={
                "time_offset": SemanticType.TIME,
                "time_cutoff": SemanticType.TIME,
            },
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="time_offset",
        )


class ProductDataset(Dataset):
    # TODO: pandas is interpreting the time_stamps wrong
    # I think its a unit issue (unix time is in secs but expected in ns)

    name = "rtb-product"

    # raw file names
    product_file_name = "All_Amazon_Meta.json"
    review_file_name = "All_Amazon_Review.json"

    # number of lines in the raw files
    product_lines = 15_023_059
    # review_lines = 233_055_327

    # for now I am playing with smaller files
    # product_lines = 1_500_000
    review_lines = 20_000_000

    # regex for parsing price
    price_re = re.compile(r"\$(\d+\.\d+)")
    price_range_re = re.compile(r"\$(\d+\.\d+) - \$(\d+\.\d+)")

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTV()}

    # TODO: implement get_cutoff_times()

    def download(self, path: str | os.PathLike) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raise NotImplementedError

    def parse_price(self, price_str: str) -> float:
        r"""Parse the raw price string into a float."""

        m = self.price_range_re.match(price_str)

        if m is not None:
            lb = float(m.group(1))
            ub = float(m.group(2))
            return (lb + ub) / 2

        m = self.price_re.match(price_str)

        if m:
            return float(m.group(1))
        else:
            raise ValueError(f"Invalid price string: {price_str}")

    def process(self) -> Database:
        r"""Process the raw files into a database."""

        # tried speeding up the json decoding with multiprocessing and others,
        # but that was just a big waste of time and gave no significant gain

        tables = {}

        # product table
        # products = []
        # path = f"{self.root}/{self.name}/raw/{self.product_file_name}"
        # with open(path, "r") as f:
        #     for l in tqdm(f, total=self.product_lines):
        #         raw = json.loads(l)
        #         try:
        #             products.append(
        #                 {
        #                     "product_id": raw["asin"],
        #                     "category": raw["category"][0],
        #                     "brand": raw["brand"],
        #                     "title": raw["title"],
        #                     "description": raw["description"],
        #                     "price": self.parse_price(raw["price"]),
        #                 }
        #             )
        #         except (ValueError, IndexError):
        #             # ignoring invalid prices and empty categories for now
        #             pass

        # tables["product"] = Table(
        #     df=pd.DataFrame(products),
        #     feat_cols={
        #         "category": SemanticType.CATEGORICAL,
        #         "brand": SemanticType.TEXT,  # can also be categorical
        #         "title": SemanticType.TEXT,
        #         "description": SemanticType.TEXT,
        #         "price": SemanticType.NUMERICAL,
        #     },
        #     fkeys={},
        #     pkey="product_id",
        #     time_col=None,
        # )

        tables["product"] = Table.load(
            f"{self.root}/{self.name}/processed/db/product.parquet"
        )

        product_ids = set(tables["product"].df["product_id"])

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
            feat_cols={
                "name": SemanticType.TEXT,
            },
            fkeys={},
            pkey="customer_id",
            time_col=None,
        )

        tables["review"] = Table(
            df=pd.DataFrame(reviews),
            feat_cols={
                "review_time": SemanticType.TIME,
                "rating": SemanticType.NUMERICAL,
                "verified": SemanticType.CATEGORICAL,
                "review_text": SemanticType.TEXT,
                "summary": SemanticType.TEXT,
            },
            fkeys={
                "customer_id": "customer",
                "product_id": "product",
            },
            pkey=None,
            time_col="review_time",
        )

        return Database(tables)

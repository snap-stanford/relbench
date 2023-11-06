import json
import multiprocessing as mp
import os
import re

# import duckdb
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

    def make_table(db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for LTV."""

        # columns in time_window_df: offset, cutoff

        product = db.tables["product"]
        review = db.tables["review"]
        table = duckdb.sql(
            r"""
            select * from product, review
            where product.product_id = review.product_id
            """
        )

        # due to query optimization and parallelization,
        # this should be fast enough
        # and doing sql queries is also flexible enough to easily implement
        # a variety of other tasks
        df = duckdb.sql(
            r"""
            select
                customer_id,
                offset,
                cutoff,
                sum(price) as ltv
            from
                table,
                sampler_df
            where
                table.time_stamp > time_window_df.offset and
                table.time_stamp <= time_window_df.cutoff and
            group by customer_id, offset, cutoff
        """
        )

        return Table(
            df=df,
            feat_cols=["offset", "cutoff"],
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="offset",
        )


price_re = re.compile(r"\$(\d+\.\d+)")
price_range_re = re.compile(r"\$(\d+\.\d+) - \$(\d+\.\d+)")


def process_price(price_str: str) -> float:
    r"""Process the raw price string into a float."""

    m = price_range_re.match(price_str)

    if m is not None:
        lb = float(m.group(1))
        ub = float(m.group(2))
        return (lb + ub) / 2

    m = price_re.match(price_str)

    if m:
        return float(m.group(1))
    else:
        raise ValueError(f"Invalid price string: {price_str}")


def decode_product_line(line):
    raw = json.loads(line)
    try:
        return {
            "category": raw["category"][0],
            "price": process_price(raw["price"]),
            "product_id": raw["asin"],
            "brand": raw["brand"],
            "title": raw["title"],
            "description": raw["description"],
        }
    except (ValueError, IndexError):
        return None


class ProductDataset(Dataset):
    name = "rtb-product"

    # raw file names
    product_file_name = "All_Amazon_Meta.json"
    review_file_name = "All_Amazon_Review.json"

    # number of lines in the raw files
    product_lines = 15_023_059
    review_lines = 233_055_327

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTV()}

    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times."""

        raise NotImplementedError

    def download(self, path: str | os.PathLike) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raise NotImplementedError

    def process_db(self) -> Database:
        r"""Process the raw files into a database."""

        mp.set_start_method("forkserver")

        tables = {}

        # product table
        path = f"{self.root}/{self.name}/raw/{self.product_file_name}"
        products = []
        num_workers = min(64, os.cpu_count())
        with mp.Pool(num_workers) as pool, open(path, "r") as f:
            products = [
                product
                for product in tqdm(
                    # determinism is not important because of temporal splits
                    pool.imap_unordered(
                        decode_product_line,
                        f,
                        chunksize=1_000,
                    ),
                    total=self.product_lines,
                )
                if product is not None
            ]

        print(len(products))
        breakpoint()

        # product table
        products = []
        skip_ctr = 0
        with open(f"{self.root}/{self.name}/raw/{self.product_file_name}", "r") as f:
            for l in tqdm(f, total=self.product_lines):
                raw = json.loads(l)
                try:
                    products.append(
                        {
                            "product_id": raw["asin"],
                            "category": raw["category"][0],
                            "brand": raw["brand"],
                            "title": raw["title"],
                            "description": raw["description"],
                            "price": self.process_price(raw["price"]),
                        }
                    )
                except (ValueError, IndexError):
                    skip_ctr += 1

        print(f"Skipped {skip_ctr} products.")

        tables["product"] = Table(
            df=pd.DataFrame(products),
            feat_cols={
                "category": SemanticType.CATEGORICAL,
                "brand": SemanticType.TEXT,  # can also be categorical
                "title": SemanticType.TEXT,
                "description": SemanticType.TEXT,
                "price": SemanticType.NUMERICAL,
            },
            fkeys={},
            pkey="product_id",
            time_col=None,
        )

        # review table
        customers = {}
        reviews = []
        with open(self.review_file_name, "r") as f:
            for l in tqdm(f, total=self.review_lines):
                raw = json.loads(l)
                customers[raw["reviewerID"]] = raw["reviewerName"]
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

        tables["customer"] = Table(
            df=pd.DataFrame(
                {
                    "customer_id": list(customers.keys()),
                    "name": list(customers.values()),
                }
            ),
            feat_cols={
                "name": TEXT,
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

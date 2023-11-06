import json
import re

import duckdb
import pandas as pd
from tqdm.auto import tqdm

import rtb


class LTV(rtb.data.Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user reviews in the time_frame."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=rtb.data.TaskType.REGRESSION,
            metrics=["mse", "smape"],
        )

    def make_table(db: rtb.data.Database, time_window_df: pd.DataFrame) -> rtb.Table:
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

        return rtb.Table(
            df=df,
            feat_cols=["offset", "cutoff"],
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="offset",
        )


class ProductDataset(rtb.data.Dataset):
    name = "rtb-product"

    # raw file names
    product_file_name = "All_Amazon_Meta.json"
    review_file_name = "All_Amazon_Review.json"

    # number of lines in the raw files
    product_lines = 15_023_059
    review_lines = 233_055_327

    price_re = re.compile(r"\$(\d+\.\d+)")
    price_range_re = re.compile(r"\$(\d+\.\d+) - \$(\d+\.\d+)")

    def __init__(self, root: str | os.PathLike) -> None:
        super().__init__(root)

        p = r"\$(\d+\.\d+)"
        self.price_re = re.compile(rf"{p}|{p} - {p}")

    def get_tasks(self) -> dict[str, rtb.data.Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTV()}

    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times."""

        raise NotImplementedError

    def download(self, path: str | os.PathLike) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raise NotImplementedError

    def process_price(self, price_str: str) -> float:
        r"""Process the raw price string into a float."""

        m = self.price_range_re.match(raw_price)

        if m is not None:
            lb = float(m.group(1))
            ub = float(m.group(2))
            return (lb + ub) / 2

        m = self.price_re.match(raw_price)
        return float(m.group(1))

    def process_db(self) -> rtb.data.Database:
        r"""Process the raw files into a database."""

        tables = {}

        # product table
        products = []
        with open(self.product_file_name, "r") as f:
            for l in tqdm(f, total=self.product_lines):
                raw = json.loads(l)
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

        tables["product"] = rtb.data.Table(
            df=pd.DataFrame(products),
            feat_cols={
                "category": rtb.data.CATEGORICAL,
                "brand": rtb.data.TEXT,  # can also be categorical
                "title": rtb.data.TEXT,
                "description": rtb.data.TEXT,
                "price": rtb.data.NUMERICAL,
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

        tables["customer"] = rtb.data.Table(
            df=pd.DataFrame(
                {
                    "customer_id": list(customers.keys()),
                    "name": list(customers.values()),
                }
            ),
            feat_cols={
                "name": rtb.data.TEXT,
            },
            fkeys={},
            pkey="customer_id",
            time_col=None,
        )

        tables["review"] = rtb.data.Table(
            df=pd.DataFrame(reviews),
            feat_cols={
                "review_time": rtb.data.TIME,
                "rating": rtb.data.NUMERICAL,
                "verified": rtb.data.CATEGORICAL,
                "review_text": rtb.data.TEXT,
                "summary": rtb.data.TEXT,
            },
            fkeys={
                "customer_id": "customer",
                "product_id": "product",
            },
            pkey=None,
            time_col="review_time",
        )

        return rtb.data.Database(tables)

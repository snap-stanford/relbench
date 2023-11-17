import os
import time
from typing import Dict, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.json
from rtb.data.database import Database
from rtb.data.dataset import Dataset
from rtb.data.table import Table
from rtb.data.task import Task, TaskType
from rtb.utils import download_url, unzip


class ChurnTask(Task):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "customer_id"],
            target_col="churn",
            task_type=TaskType.BINARY_CLASSIFICATION,
            window_sizes=[pd.Timedelta("52W")],
            metrics=["auprc"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        customer = db.tables["customer"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            """
            SELECT
                window_min_time,
                window_max_time,
                customer_id,
                NOT EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.review_time BETWEEN window_min_time AND window_max_time
                ) AS churn
            FROM
                time_window_df,
                customer
        """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="window_min_time",
        )


class LTVTask(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "customer_id"],
            target_col="ltv",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta("52W")],
            metrics=["auprc"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        customer = db.tables["customer"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            """
            SELECT
                window_min_time,
                window_max_time,
                customer_id,
                ltv,
                count
            FROM
                time_window_df,
                customer,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as ltv,
                        COALESCE(COUNT(price), 0) as count
                    FROM review, product
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.product_id = product.product_id AND
                        review.review_time BETWEEN window_min_time AND window_max_time
                )
        """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="window_min_time",
        )


class ProductDataset(Dataset):
    name = "rtb-product"

    cat_to_raw = {
        "books": "Books",
        "fashion": "AMAZON_FASHION",
    }

    def __init__(
        self,
        root: Union[str, os.PathLike],
        process=False,
        category: str = "books",
        use_5_core: bool = True,
    ):
        self.category = category
        self.use_5_core = use_5_core

        self.name = f"{self.__class__.name}/{self.category}{'-5-core' if self.use_5_core else ''}"

        super().__init__(root, process)

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTVTask(), "churn": ChurnTask()}

    # TODO: implement get_cutoff_times()

    def download_raw(self, path: Union[str, os.PathLike]) -> None:
        r"""Download the Amazon dataset raw files from the AWS server and
        decompresses it."""

        raw = self.cat_to_raw[self.category]

        # download review file
        if self.use_5_core:
            url = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/{raw}_5.json.gz"
        else:
            url = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/{raw}.json.gz"

        download_path = download_url(url, path)
        # TODO: this doesn't work, need gunzip
        unzip(download_path, path)

        # download product file
        url = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/meta_{raw}.json.gz"
        download_path = download_url(url, path)
        # TODO: this doesn't work, need gunzip
        unzip(download_path, path)

    def process(self) -> Database:
        r"""Process the raw files into a database."""

        ### product table ###

        file_name = f"meta_{self.cat_to_raw[self.category]}.json"
        path = f"{self.root}/{self.name}/raw/{file_name}"
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

        print("converting to pandas dataframe...")
        tic = time.time()
        pdf = ptable.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print("processing product info...")
        tic = time.time()

        # asin is not intuitive / recognizable
        pdf.rename(columns={"asin": "product_id"}, inplace=True)

        # somehow the raw data has duplicate product_id's
        pdf.drop_duplicates(subset=["product_id"], inplace=True)

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

        file_name = (
            f"{self.cat_to_raw[self.category]}{'_5' if self.use_5_core else ''}.json"
        )
        path = f"{self.root}/{self.name}/raw/{file_name}"
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

        print("converting to pandas dataframe...")
        tic = time.time()
        rdf = rtable.to_pandas()
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print("processing review and customer info...")
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
        pdf.query("product_id == @plist", inplace=True)
        rdf.query("product_id == @plist", inplace=True)
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
                    fkey_col_to_pkey_table={},
                    pkey_col="product_id",
                    time_col=None,
                ),
                "customer": Table(
                    df=cdf,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
                    time_col=None,
                ),
                "review": Table(
                    df=rdf,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                    pkey_col=None,
                    time_col="review_time",
                ),
            }
        )

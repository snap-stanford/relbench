import os
import time
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.json

from rtb.data import Database, RelBenchDataset, Table
from rtb.tasks.amazon_reviews import CustomerChurnTask, CustomerLTVTask


class AmazonReviewsDataset(RelBenchDataset):
    name = "amazon_reviews"
    val_timestamp = pd.Timestamp("2013-01-01")
    test_timestamp = pd.Timestamp("2015-01-01")
    task_cls_list = [CustomerChurnTask, CustomerLTVTask]

    category_list = ["books", "fashion"]

    url_prefix = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2"
    _category_to_url_key = {"books": "Books", "fashion": "AMAZON_FASHION"}

    def __init__(
        self,
        root: Union[str, os.PathLike],
        category: str = "books",
        use_5_core: bool = True,
        *,
        download=False,
        process=False,
    ):
        self.category = category
        self.use_5_core = use_5_core

        self.name = f"{self.name}-{category}{'_5_core' if use_5_core else ''}"

        super().__init__(root, download=download, process=process)

    def download_raw_db(self, raw_path: Union[str, os.PathLike]) -> None:
        url_key = self._category_to_url_key[self.category]

        # download review file
        if self.use_5_core:
            url = f"{self.url_prefix}/categoryFilesSmall/{url_key}_5.json.gz"
        else:
            url = f"{self.url_prefix}/categoryFiles/{url_key}.json.gz"

        download_and_extract(url, raw_path)

        # download product file
        url = f"{self.url_prefix}/categoryFilesSmall/meta_{url_key}.json.gz"
        download_and_extract(url, raw_path)

    def make_db(self, raw_path: Union[str, os.PathLike]) -> Database:
        r"""Process the raw files into a database."""

        ### product table ###

        file_name = f"meta_{self._cat_to_raw[self.category]}.json"
        path = raw_path / file_name
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
            f"{self._cat_to_raw[self.category]}{'_5' if self.use_5_core else ''}.json"
        )
        path = raw_path / file_name
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

        print("keeping only products common to product and review tables...")
        tic = time.time()
        plist = list(set(pdf["product_id"]) & set(rdf["product_id"]))
        pdf.query("product_id == @plist", inplace=True)
        rdf.query("product_id == @plist", inplace=True)
        toc = time.time()
        print(f"done in {toc - tic:.2f} seconds.")

        print("extracting customer table...")
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
            table_dict={
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

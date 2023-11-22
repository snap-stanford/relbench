import os
import random
import string
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from rtb.data.database import Database
from rtb.data.dataset import Dataset
from rtb.data.table import Table
from rtb.data.task import Task
from rtb.datasets.product import ChurnTask, LTVTask


def _generate_random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(length))
    return random_string


class FakeProductDataset(Dataset):
    r"""Fake e-commerce dataset for testing purposes. Schema is similar to
    ProductDataset."""

    name = "rtb-fake-product"

    def get_tasks(self) -> Dict[str, Task]:
        return {"ltv": LTVTask(), "churn": ChurnTask()}

    def download_raw(self, path: Union[str, os.PathLike]) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        return

    def download_processed(self, path: Union[str, os.PathLike]) -> None:
        raise RuntimeError(
            "download_processed not supported for"
            " FakeProductDataset. Use process=True to force"
            " processing for the first use."
        )

    def process(self) -> Database:
        num_products = 30
        num_customers = 100
        num_transactions = 500

        product_df = pd.DataFrame(
            {
                "product_id": [f"product_id_{i}" for i in range(num_products)],
                "category": [None, [], ["toy", "health"]] * (num_products // 3),
                "title": [_generate_random_string(5, 15) for _ in range(num_products)],
                "price": np.random.rand(num_products) * 10,
            }
        )
        customer_df = pd.DataFrame(
            {
                "customer_id": [f"customer_id_{i}" for i in range(num_customers)],
                "age": np.random.randint(10, 50, size=(num_customers,)),
                "gender": ["male", "female"] * (num_customers // 2),
            }
        )
        # Add some dangling foreign keys:
        review_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_transactions)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_transactions)
                ],
                "review_time": pd.to_datetime(
                    10 * np.arange(num_transactions), unit="D"
                ),
                "rating": np.random.randint(1, 6, size=(num_transactions,)),
            }
        )

        tables: Dict[str, Table] = {}

        tables["product"] = Table(
            df=product_df,
            fkey_col_to_pkey_table={},
            pkey_col="product_id",
        )
        tables["customer"] = Table(
            df=customer_df,
            fkey_col_to_pkey_table={},
            pkey_col="customer_id",
        )
        tables["review"] = Table(
            df=review_df,
            fkey_col_to_pkey_table={
                "customer_id": "customer",
                "product_id": "product",
            },
            time_col="review_time",
        )

        return Database(tables)

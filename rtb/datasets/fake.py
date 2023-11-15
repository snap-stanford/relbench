import os
from typing import Any, Dict, Union

import random
import duckdb
import numpy as np
import pandas as pd
from rtb.data.database import Database
from rtb.data.dataset import Dataset
from rtb.data.table import Table
from rtb.data.task import Task, TaskType


class LTV(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user has made transactions in a given time frame."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=TaskType.REGRESSION,
            test_time_window_sizes=[pd.Timedelta("1W")],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for LTV."""
        product = db.tables["product"].df
        transaction = db.tables["transaction"].df

        # due to query optimization and parallelization,
        # this should be fast enough
        # and doing sql queries is also flexible enough to easily implement
        # a variety of other tasks
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
                        timestamp,
                        customer_id,
                        price
                    FROM
                        product,
                        transaction
                    WHERE
                        product.product_id = transaction.product_id
                ) AS tmp
            WHERE
                tmp.timestamp > time_window_df.window_min_time AND
                tmp.timestamp <= time_window_df.window_max_time
            GROUP BY customer_id, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="window_min_time",
        )


class FakeEcommerceDataset(Dataset):
    r"""Fake e-commerce dataset for testing purposes."""

    name = "rtb-fake-ecommerce"

    def __init__(self, root: Union[str, os.PathLike], process: bool = False) -> None:
        super().__init__(root, process)

    def get_tasks(self) -> Dict[str, Task]:
        return {"ltv": LTV()}

    def download(self, url: str, path: Union[str, os.PathLike]) -> None:
        pass

    def process(self) -> Database:
        num_products = 30
        num_customers = 100
        num_transactions = 500

        product_df = pd.DataFrame(
            {
                "product_id": [f"product_id_{i}" for i in range(num_products)],
                "category": ["toy", "health", "digital"] * (num_products // 3),
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
        transaction_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_transactions)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_transactions)
                ],
                "timestamp": pd.to_datetime(10 * np.arange(num_transactions), unit="d"),
                "price": np.random.rand(num_transactions) * 10,
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
        tables["transaction"] = Table(
            df=transaction_df,
            fkey_col_to_pkey_table={
                "customer_id": "customer",
                "product_id": "product",
            },
            time_col="timestamp",
        )

        return Database(tables)

    def get_stype_proposal(self) -> Dict[str, Dict[str, Any]]:
        from torch_frame import stype

        stype_dict: Dict[str, Dict[str, Any]] = {}
        stype_dict["product"] = {
            "category": stype.categorical,
        }
        stype_dict["customer"] = {
            "age": stype.numerical,
            "gender": stype.categorical,
        }
        stype_dict["transaction"] = {
            # TODO: add when timestamp gets supported in torch-frame
            # "timestamp": stype.timestamp,
            "price": stype.numerical,
        }

        return stype_dict

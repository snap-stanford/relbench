import os
import numpy as np

from typing import Dict
import duckdb
import pandas as pd
from torch_frame import stype

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset


class LTV(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user has made transactions in the time_frame."""

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
            fkeys={"customer_id": "customer"},
            pkey_col=None,
            time_col="window_min_time",
        )


class FakeEcommerceDataset(Dataset):
    r"""Fake e-commerce dataset for the testing purpose."""

    name = "rtb-fake-ecommerce"

    def __init__(self, root: str | os.PathLike, process=False) -> None:
        super().__init__(root, process)
        col_to_stype_dict = {}
        col_to_stype_dict["product"] = {"category": stype.categorical}
        col_to_stype_dict["customer"] = {
            "age": stype.numerical,
            "gender": stype.categorical,
        }
        col_to_stype_dict["transaction"] = {
            # TODO: add back when timestamp gets supported in torch-frame
            # "timestamp": stype.timestamp,
            "price": stype.numerical,
        }
        self._col_to_stype_dict = col_to_stype_dict

    def get_tasks(self) -> Dict[str, Task]:
        return {"ltv": LTV()}

    def download(self, url: str, path: str | os.PathLike) -> None:
        pass

    def process(self) -> Database:
        num_products = 30
        num_customers = 100
        num_transactions = 500

        product_df = pd.DataFrame(
            {
                "product_id": np.arange(num_products),
                "category": ["toy", "health", "digital"] * (num_products // 3),
            }
        )
        customer_df = pd.DataFrame(
            {
                "customer_id": np.arange(num_customers),
                "age": np.random.randint(10, 50, size=(num_customers,)),
                "gender": ["male", "female"] * (num_customers // 2),
            }
        )
        transaction_df = pd.DataFrame(
            {
                "customer_id": np.random.randint(
                    0, num_customers, size=(num_transactions,)
                ),
                "product_id": np.random.randint(
                    0, num_products, size=(num_transactions,)
                ),
                "timestamp": pd.to_datetime(10 * np.arange(num_transactions), unit="d"),
                "price": np.random.rand(num_transactions) * 10,
            }
        )
        return Database(
            tables={
                "product": Table(
                    df=product_df,
                    fkeys={},
                    pkey_col="product_id",
                    time_col=None,
                ),
                "customer": Table(
                    df=customer_df,
                    fkeys={},
                    pkey_col="customer_id",
                    time_col=None,
                ),
                "transaction": Table(
                    df=transaction_df,
                    fkeys={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                    pkey_col=None,
                    time_col="timestamp",
                ),
            }
        )

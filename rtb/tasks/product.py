import os
import time
from typing import Dict, Union

import duckdb
import pandas as pd
import pyarrow as pa

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

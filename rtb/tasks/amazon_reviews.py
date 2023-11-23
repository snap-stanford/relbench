import os
import time
from typing import Dict, Union

import duckdb
import pandas as pd
import pyarrow as pa

from rtb.data import Database, Dataset, Table, Task, TaskType
from rtb.metrics import accuracy, f1, mae, rmse, roc_auc
from rtb.utils import download_url, unzip


class CustomerChurnTask(Task):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    input_cols = ["timestamp", "timedelta", "customer_id"]
    target_col = "churn"
    task_type = TaskType.BINARY_CLASSIFICATION
    benchmark_timedelta_list = [pd.Timedelta(days=365), pd.Timedelta(days=365 * 2)]
    benchmark_metric_dict = {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}

    @classmethod
    def make_table(cls, db: Database, time_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        customer = db.tables["customer"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            """
            SELECT
                timestamp,
                timedelta,
                customer_id,
                NOT EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.review_time > timestamp AND
                        review.review_time <= timestamp + timedelta
                ) AS churn
            FROM
                time_df,
                customer
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )


class CustomerLTVTask(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    input_cols = ["timestamp", "timedelta", "customer_id"]
    target_col = "ltv"
    task_type = TaskType.REGRESSION
    benchmark_timedelta_list = [pd.Timedelta(days=365), pd.Timedelta(days=365 * 2)]
    benchmark_metric_dict = {"mae": mae, "rmse": rmse}

    @classmethod
    def make_table(cls, db: Database, time_df: pd.DataFrame) -> Table:
        product = db.tables["product"].df
        customer = db.tables["customer"].df
        review = db.tables["review"].df

        df = duckdb.sql(
            """
            SELECT
                timestamp,
                timedelta,
                customer_id,
                ltv,
                count
            FROM
                time_df,
                customer,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as ltv,
                        COALESCE(COUNT(price), 0) as count
                    FROM review, product
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.product_id = product.product_id AND
                        review.review_time > timestamp AND
                        reviewear
                        review_time <= timestamp + timedelta
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )

import os
import time
from typing import Dict, Union

import duckdb
import pandas as pd

from rtb.data import Database, Dataset, Table, Task
from rtb.metrics import accuracy, f1, mae, rmse, roc_auc


class CustomerChurnTask(Task):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    name = "customer_churn"

    task_type = "binary_classification"
    metrics = [accuracy, f1, roc_auc]

    timedelta = pd.Timedelta(days=365 * 2)

    target_col = "churn"
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"

    @classmethod
    def make_table(cls, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review
                        WHERE
                            review.customer_id = customer.customer_id AND
                            review_time > timestamp AND
                            review_time <= timestamp + INTERVAL '{cls.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp - INTERVAL '{cls.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={cls.entity_col: cls.entity_table},
            pkey_col=None,
            time_col=cls.time_col,
        )


class CustomerLTVTask(Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    name = "customer_ltv"

    task_type = "regression"
    metrics = [mae, rmse]

    timedelta = pd.Timedelta(days=365 * 2)

    target_col = "ltv"
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"

    @classmethod
    def make_table(cls, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                ltv,
                count_
            FROM
                timestamp_df,
                customer,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as ltv,
                        COALESCE(COUNT(price), 0) as count_
                    FROM
                        review,
                        product
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.product_id = product.product_id AND
                        review_time > timestamp AND
                        review_time <= timestamp + INTERVAL '{cls.timedelta}'
                )
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp - INTERVAL '{cls.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )

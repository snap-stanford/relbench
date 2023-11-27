import os

import duckdb
import pandas as pd

from relbench.data import Database, RelBenchTask, Table
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc


class ChurnTask(RelBenchTask):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    name = "rel-amazon-churn"
    task_type = "binary_classification"
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
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
                            review_time <= timestamp + INTERVAL '{self.timedelta}'
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
                        review_time > timestamp - INTERVAL '{self.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class LTVTask(RelBenchTask):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    name = "rel-amazon-ltv"
    task_type = "regression"
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
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
                        review_time <= timestamp + INTERVAL '{self.timedelta}'
                )
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp - INTERVAL '{self.timedelta}' AND
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

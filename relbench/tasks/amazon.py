import duckdb
import pandas as pd

from relbench.data import Database, LinkTask, NodeTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class UserChurnTask(NodeTask):
    r"""Churn for a customer is 1 if the customer does not review any product
    in the time window, else 0."""

    name = "user-churn"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
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


class UserLTVTask(NodeTask):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the customer reviews in the time window."""

    name = "user-ltv"
    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]

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
            FROM
                timestamp_df,
                customer,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as ltv,
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


class ItemChurnTask(NodeTask):
    r"""Churn for a product is 1 if the product recieves at least one review
    in the time window, else 0."""

    name = "item-churn"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
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
                product_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review
                        WHERE
                            review.product_id = product.product_id AND
                            review_time > timestamp AND
                            review_time <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                product,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.product_id = product.product_id AND
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


class ItemLTVTask(NodeTask):
    r"""LTV (life-time value) for a product is the numer of times the product
    is purchased in the time window multiplied by price."""

    name = "item-ltv"
    task_type = TaskType.REGRESSION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                product.product_id,
                COALESCE(SUM(price), 0) AS ltv,
            FROM
                timestamp_df,
                product,
                review
            WHERE
                review.product_id = product.product_id AND
                review_time > timestamp AND
                review_time <= timestamp + INTERVAL '{self.timedelta}'
            GROUP BY
                timestamp,
                product.product_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col="timestamp",
        )


class UserItemPurchaseTask(LinkTask):
    r"""Predict the list of distinct items each customer will purchase in the
    next two years."""

    name = "user-item-purchase"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                review.customer_id,
                LIST(DISTINCT review.product_id) AS product_id
            FROM
                timestamp_df t
            LEFT JOIN
                review
            ON
                review.review_time > t.timestamp AND
                review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                review.customer_id is not null and review.product_id is not null
            GROUP BY
                t.timestamp,
                review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserItemRateTask(LinkTask):
    r"""Predict the list of distinct items each customer will purchase and give a 5 star review in the
    next two years."""

    name = "user-item-rate"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    review.customer_id,
                    LIST(DISTINCT review.product_id) AS product_id
                FROM
                    timestamp_df t
                LEFT JOIN
                    review
                ON
                    review.review_time > t.timestamp AND
                    review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
                WHERE
                    review.customer_id IS NOT NULL
                    AND review.product_id IS NOT NULL
                    AND review.rating = 5.0
                GROUP BY
                    t.timestamp,
                    review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserItemReviewTask(LinkTask):
    r"""Predict the list of distinct items each customer will purchase and give a detailed review in the
    next two years."""

    name = "user-item-review"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        REVIEW_LENGTH = (
            300  # minimum length of review to be considered as detailed review
        )

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    review.customer_id,
                    LIST(DISTINCT review.product_id) AS product_id
                FROM
                    timestamp_df t
                LEFT JOIN
                    review
                ON
                    review.review_time > t.timestamp AND
                    review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
                WHERE
                    review.customer_id IS NOT NULL
                    AND review.product_id IS NOT NULL
                    AND (LENGTH(review.review_text) > {REVIEW_LENGTH} AND review.review_text IS NOT NULL)
                GROUP BY
                    t.timestamp,
                    review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

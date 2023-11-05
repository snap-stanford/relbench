import duckdb
import pandas as pd

import rtb


class LTV(rtb.data.Task):
    r"""LTV (life-time value) for a customer is the sum of prices of products
    that the user reviews in the time_frame."""

    def __init__(self):
        super().__init__(
            target_col="ltv",
            task_type=rtb.data.TaskType.REGRESSION,
            metrics=["mse", "smape"],
        )

    def make_table(db: rtb.data.Database, time_window_df: pd.DataFrame) -> rtb.Table:
        r"""Create Task object for LTV."""

        # columns in time_window_df: offset, cutoff

        product = db.tables["product"]
        review = db.tables["review"]
        table = duckdb.sql(
            r"""
            select * from product, review
            where product.product_id = review.product_id
            """
        )

        # due to query optimization and parallelization,
        # this should be fast enough
        # and doing sql queries is also flexible enough to easily implement
        # a variety of other tasks
        df = duckdb.sql(
            r"""
            select
                customer_id,
                offset,
                cutoff,
                sum(price) as ltv
            from
                table,
                sampler_df
            where
                table.time_stamp > time_window_df.offset and
                table.time_stamp <= time_window_df.cutoff and
            group by customer_id, offset, cutoff
        """
        )

        return rtb.Table(
            df=df,
            feat_cols=["offset", "cutoff"],
            fkeys={"customer_id": "customer"},
            pkey=None,
            time_col="offset",
        )


class ProductDataset(rtb.data.Dataset):
    name = "mtb-product"

    def get_tasks(self) -> dict[str, rtb.data.Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"ltv": LTV()}

    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times."""

        raise NotImplementedError

    def download(self) -> None:
        r"""Download the Amazon dataset raw files from the AWS server."""

        raise NotImplementedError

    def process_db(self) -> rtb.data.Database:
        r"""Process the raw files into a database."""

        raise NotImplementedError

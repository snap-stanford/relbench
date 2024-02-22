import duckdb
import pandas as pd

from relbench.data import Database, RelBenchLinkTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


class RecommendationTask(RelBenchLinkTask):
    r"""Predict the list of articles each customer will purchase in the next
    seven days"""

    name = "rel-hm-rec"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customer_id,
                LIST(DISTINCT transactions.article_id) AS article_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.t_dat > t.timestamp AND
                transactions.t_dat <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                transactions.customer_id
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

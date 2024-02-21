import os
import shutil

from pathlib import Path
import pandas as pd

from relbench.data import Database, Dataset, Table
from relbench.tasks.amazon import RecommendationTask
from relbench.utils import unzip_processor


class HMDataset(Dataset):
    name = "hm-recommendation"
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )

    def __init__(self):
        db = self.make_db()
        db.reindex_pkeys_and_fkeys()
        # Set to end date right now.#
        val_timestamp = db.max_timestamp
        test_timestamp = db.max_timestamp
        max_eval_time_frames = 1
        super().__init__(
            db=db,
            val_timestamp=val_timestamp,
            test_timestamp=test_timestamp,
            max_eval_time_frames=max_eval_time_frames,
            task_cls_list=[RecommendationTask],
        )

    def make_db(self) -> Database:
        path = os.path.join("data", "hm-recommendation")
        zip = os.path.join(path, "h-and-m-personalized-fashion-recommendations.zip")
        customers = os.path.join(path, "customers.csv")
        articles = os.path.join(path, "articles.csv")
        transactions = os.path.join(path, "transactions_train.csv")
        hold_out = os.path.join(path, "sample_submission.csv")
        if not os.path.exists(customers):
            if not os.path.exists(zip):
                raise RuntimeError(
                    f"Dataset not found. Please download "
                    f"h-and-m-personalized-fashion-recommendations.zip from "
                    f"'{self.url}' and move it to '{path}'"
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)

        articles_df = pd.read_csv(articles)
        customers_df = pd.read_csv(customers)
        transactions_df = pd.read_csv(transactions)

        return Database(
            table_dict={
                "article": Table(
                    df=articles_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="article_id",
                ),
                "customer": Table(
                    df=customers_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
                ),
                "transactions": Table(
                    df=transactions_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "article_id": "article",
                    },
                    time_col="t_dat",
                ),
            }
        )
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
from torch_frame import stype

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.hm import ItemSalesTask, UserChurnTask, UserItemPurchaseTask


class HMDataset(RelBenchDataset):
    name = "rel-hm"
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )
    # Train for the most recent 1 year out of 2 years of the original
    # time period
    train_start_timestamp = pd.Timestamp("2019-09-07")
    val_timestamp = pd.Timestamp("2020-09-07")
    test_timestamp = pd.Timestamp("2020-09-14")
    max_eval_time_frames = 1
    task_cls_list = [UserItemPurchaseTask, UserChurnTask, ItemSalesTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        path = os.path.join("data", "hm-recommendation")
        zip = os.path.join(path, "h-and-m-personalized-fashion-recommendations.zip")
        customers = os.path.join(path, "customers.csv")
        articles = os.path.join(path, "articles.csv")
        transactions = os.path.join(path, "transactions_train.csv")
        if not os.path.exists(customers):
            if not os.path.exists(zip):
                raise RuntimeError(
                    f"Dataset not found. Please download "
                    f"h-and-m-personalized-fashion-recommendations.zip from "
                    f"'{self.url}' and move it to '{path}'. Once you have your"
                    f"Kaggle API key, you can use the following command: "
                    f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations"
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)

        articles_df = pd.read_csv(articles)
        customers_df = pd.read_csv(customers)
        transactions_df = pd.read_csv(transactions)
        transactions_df["t_dat"] = pd.to_datetime(
            transactions_df["t_dat"], format="%Y-%m-%d"
        )

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

    @property
    def col_to_stype_dict(self) -> dict[str, dict[str, stype]]:
        return {
            "article": {
                "article_id": stype.numerical,
                "product_code": stype.numerical,
                "prod_name": stype.text_embedded,
                "product_type_no": stype.numerical,
                "product_type_name": stype.categorical,
                "product_group_name": stype.categorical,
                "graphical_appearance_no": stype.categorical,
                "graphical_appearance_name": stype.categorical,
                "colour_group_code": stype.categorical,
                "colour_group_name": stype.categorical,
                "perceived_colour_value_id": stype.categorical,
                "perceived_colour_value_name": stype.categorical,
                "perceived_colour_master_id": stype.numerical,
                "perceived_colour_master_name": stype.categorical,
                "department_no": stype.numerical,
                "department_name": stype.categorical,
                "index_code": stype.categorical,
                "index_name": stype.categorical,
                "index_group_no": stype.categorical,
                "index_group_name": stype.categorical,
                "section_no": stype.numerical,
                "section_name": stype.text_embedded,
                "garment_group_no": stype.categorical,
                "garment_group_name": stype.categorical,
                "detail_desc": stype.text_embedded,
            },
            "customer": {
                "customer_id": stype.text_embedded,
                "FN": stype.categorical,
                "Active": stype.categorical,
                "club_member_status": stype.categorical,
                "fashion_news_frequency": stype.categorical,
                "age": stype.numerical,
                "postal_code": stype.categorical,
            },
            "transactions": {
                "t_dat": stype.timestamp,
                "price": stype.numerical,
                "sales_channel_id": stype.categorical,
            },
        }

import os

import pandas as pd

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.avito import UserAdClickTask, UserClicksTask
from relbench.utils import clean_datetime


class AvitoDataset(RelBenchDataset):
    name = "rel-avito"
    url = "https://www.kaggle.com/competitions/avito-context-ad-clicks"

    # search stream ranges from 2015-04-25 to 2015-05-20
    train_start_timestamp = pd.Timestamp("2015-04-25")
    val_timestamp = pd.Timestamp("2015-05-09")
    test_timestamp = pd.Timestamp("2015-05-14")
    max_eval_time_frames = 1
    task_cls_list = [UserClicksTask, UserAdClickTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        # Customize path as necessary
        path = os.path.join("data", "avito_integ_test")

        # Define table names
        ads_info = os.path.join(path, "AdsInfo")
        category = os.path.join(path, "Category")
        location = os.path.join(path, "Location")
        phone_requests_stream = os.path.join(path, "PhoneRequestsStream")
        search_info = os.path.join(path, "SearchInfo")
        search_stream = os.path.join(path, "SearchStream")
        user_info = os.path.join(path, "UserInfo")
        visit_stream = os.path.join(path, "VisitStream")

        # Load table as pandas dataframes
        ads_info_df = pd.read_parquet(ads_info)
        ads_info_df.dropna(subset=["AdID"], inplace=True)
        # Params column contains a dictionary of type Dict[int, str].
        # Drop it for now since we can not handle this column type yet.
        ads_info_df.drop(columns=["Params"], inplace=True)
        ads_info_df["Title"].fillna("", inplace=True)
        category_df = pd.read_parquet(category)
        location_df = pd.read_parquet(location)
        location_df.dropna(subset=["LocationID"], inplace=True)
        phone_requests_stream_df = pd.read_parquet(phone_requests_stream)
        search_info_df = pd.read_parquet(search_info)
        # SearchParams column contains a dictionary of type Dict[int, str].
        # Drop it for now since we can not handle this column type yet.
        search_info_df.drop(columns=["SearchParams"], inplace=True)
        search_stream_df = pd.read_parquet(search_stream)
        user_info_df = pd.read_parquet(user_info)
        visit_stream_df = pd.read_parquet(visit_stream)
        search_info_df = clean_datetime(search_info_df, "SearchDate")
        search_stream_df = clean_datetime(search_stream_df, "SearchDate")
        phone_requests_stream_df = clean_datetime(
            phone_requests_stream_df, "PhoneRequestDate"
        )
        visit_stream_df = clean_datetime(visit_stream_df, "ViewDate")

        tables = {}
        tables["AdsInfo"] = Table(
            df=ads_info_df,
            fkey_col_to_pkey_table={
                "LocationID": "Location",
                "CategoryID": "Category",
            },
            pkey_col="AdID",
        )
        tables["Category"] = Table(
            df=category_df,
            fkey_col_to_pkey_table={},
            pkey_col="CategoryID",
        )
        tables["Location"] = Table(
            df=location_df,
            fkey_col_to_pkey_table={},
            pkey_col="LocationID",
        )
        tables["PhoneRequestsStream"] = Table(
            df=phone_requests_stream_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "AdID": "AdsInfo",
            },
            time_col="PhoneRequestDate",
        )
        tables["SearchInfo"] = Table(
            df=search_info_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "LocationID": "Location",
                "CategoryID": "Category",
            },
            pkey_col="SearchID",
            time_col="SearchDate",
        )
        tables["SearchStream"] = Table(
            df=search_stream_df,
            fkey_col_to_pkey_table={
                "SearchID": "SearchInfo",
                "AdID": "AdsInfo",
            },
            time_col="SearchDate",
        )
        tables["UserInfo"] = Table(
            df=user_info_df,
            fkey_col_to_pkey_table={},
            pkey_col="UserID",
        )
        tables["VisitStream"] = Table(
            df=visit_stream_df,
            fkey_col_to_pkey_table={
                "UserID": "UserInfo",
                "AdID": "AdsInfo",
            },
            time_col="ViewDate",
        )
        return Database(tables)

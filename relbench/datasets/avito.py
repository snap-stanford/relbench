import os

import pandas as pd
import pooch

from relbench.data import Database, Dataset, Table
from relbench.utils import clean_datetime, unzip_processor


class AvitoDataset(Dataset):
    name = "rel-avito"
    url = "https://www.kaggle.com/competitions/avito-context-ad-clicks"
    err_msg = (
        "{data} not found. Please download avito data from "
        "'{url}' and move it to '{path}'."
    )

    # search stream ranges from 2015-04-25 to 2015-05-20
    val_timestamp = pd.Timestamp("2015-05-08")
    test_timestamp = pd.Timestamp("2015-05-14")
    max_eval_time_frames = 1

    def make_db(self) -> Database:
        # Customize path as necessary
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/rel-avito-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="24ae408ee546cf9171742288d1ec6c52e60d332dd47f58eb78fabc64a3034f43",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "avito_500k_integ_test")

        # Define table names
        ads_info = os.path.join(path, "AdsInfo")
        category = os.path.join(path, "Category")
        location = os.path.join(path, "Location")
        phone_requests_stream = os.path.join(path, "PhoneRequestsStream")
        search_info = os.path.join(path, "SearchInfo")
        search_stream = os.path.join(path, "SearchStream")
        user_info = os.path.join(path, "UserInfo")
        visit_stream = os.path.join(path, "VisitStream")
        if not os.path.exists(ads_info):
            raise RuntimeError(
                self.err_msg.format(data="Dataset", url=self.url, path=path)
            )

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
        db = Database(tables)

        db = db.from_(pd.Timestamp("2015-04-25"))

        return db

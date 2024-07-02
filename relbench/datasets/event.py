import os
import shutil
from pathlib import Path

import pandas as pd
import pooch

from relbench.data import Database, Dataset, Table
from relbench.utils import decompress_gz_file, unzip_processor


class EventDataset(Dataset):
    name = "rel-event"
    url = "https://www.kaggle.com/competitions/event-recommendation-engine-challenge"  # noqa
    err_msg = (
        "{data} not found. Please download "
        "event-recommendation-engine-challenge.zip from "
        "'{url}' and move it to '{path}'. Once you have your"
        "Kaggle API key, you can use the following command: "
        "kaggle competitions download -c event-recommendation-engine-challenge"
    )

    val_timestamp = pd.Timestamp("2012-11-21")
    test_timestamp = pd.Timestamp("2012-11-29")
    max_eval_time_frames = 1

    def check_table_and_decompress_if_exists(self, table_path: str, alt_path: str = ""):
        if not os.path.exists(table_path) or (
            alt_path != "" and not os.path.exists(alt_path)
        ):
            if os.path.exists(table_path + ".gz"):
                decompress_gz_file(table_path + ".gz", table_path)
            else:
                self.err_msg.format(data=table_path, url=self.url, path=table_path)

    def make_db(self) -> Database:
        url = "https://relbench.stanford.edu/data/rel-event-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="9cb01d6e5e8bd60db61c769656d69bdd0864ed8030d9932784e8338ed5d1183e",
            progressbar=True,
            processor=unzip_processor,
        )
        users_df = pd.read_csv(
            os.path.join(path, "users.csv"), parse_dates=["joinedAt"]
        )
        friends_df = pd.read_csv(
            os.path.join(path, "users.csv"), parse_dates=["joinedAt"]
        )
        user_friends_df = pd.read_csv(os.path.join(path, "user_friends.csv"))
        events_df = pd.read_csv(os.path.join(path, "events.csv"))
        events_df = events_df.dropna()
        events_df["user_id"] = events_df["user_id"].astype(int)
        event_attendees_df = pd.read_csv(os.path.join(path, "event_attendees.csv"))
        event_interest_df = pd.read_csv(os.path.join(path, "train.csv"))
        users_df["joinedAt"] = pd.to_datetime(
            users_df["joinedAt"], errors="coerce"
        ).dt.tz_localize(None)
        users_df["birthyear"] = pd.to_numeric(users_df["birthyear"], errors="coerce")
        friends_df["joinedAt"] = pd.to_datetime(
            friends_df["joinedAt"], errors="coerce"
        ).dt.tz_localize(None)
        friends_df["birthyear"] = pd.to_numeric(
            friends_df["birthyear"], errors="coerce"
        )
        events_df["start_time"] = pd.to_datetime(
            events_df["start_time"], errors="coerce"
        ).dt.tz_localize(None)

        event_interest_df["timestamp"] = pd.to_datetime(
            event_interest_df["timestamp"], errors="coerce"
        ).dt.tz_localize(None)
        event_attendees_df["start_time"] = pd.to_datetime(
            event_attendees_df["start_time"], errors="coerce"
        )
        event_attendees_df["start_time"] = (
            event_attendees_df["start_time"].dt.tz_localize(None).apply(pd.Timestamp)
        )

        db = Database(
            table_dict={
                "users": Table(
                    df=users_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="user_id",
                    time_col="joinedAt",
                ),
                "friends": Table(
                    df=friends_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="user_id",
                    time_col="joinedAt",
                ),
                "events": Table(
                    df=events_df,
                    fkey_col_to_pkey_table={"user_id": "friends"},
                    pkey_col="event_id",
                    time_col="start_time",
                ),
                "event_attendees": Table(
                    df=event_attendees_df,
                    fkey_col_to_pkey_table={
                        "event": "events",
                        "user_id": "users",
                    },
                    time_col="start_time",
                ),
                "event_interest": Table(
                    df=event_interest_df,
                    fkey_col_to_pkey_table={
                        "event": "events",
                        "user": "users",
                    },
                    time_col="timestamp",
                ),
                "user_friends": Table(
                    df=user_friends_df,
                    fkey_col_to_pkey_table={
                        "user": "users",
                        "friend": "friends",
                    },
                ),
            }
        )

        db = db.from_(pd.Timestamp("2012-06-20"))

        return db

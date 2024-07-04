import os

import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
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
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "rel-event")
        users = os.path.join(path, "users.csv")
        user_friends = os.path.join(path, "user_friends.csv")
        events = os.path.join(path, "events.csv")
        event_attendees = os.path.join(path, "event_attendees.csv")
        if not (os.path.exists(users)):
            if not os.path.exists(zip):
                raise RuntimeError(
                    self.err_msg.format(data="Dataset", url=self.url, path=zip)
                )
            else:
                shutil.unpack_archive(zip, Path(zip).parent)
        self.check_table_and_decompress_if_exists(
            user_friends, os.path.join(path, "user_friends_flattened.csv")
        )
        self.check_table_and_decompress_if_exists(events)
        self.check_table_and_decompress_if_exists(
            event_attendees, os.path.join(path, "event_attendees_flattened.csv")
        )
        users_df = pd.read_csv(users, dtype={"user_id": int}, parse_dates=["joinedAt"])
        users_df["birthyear"] = pd.to_numeric(users_df["birthyear"], errors="coerce")
        users_df["joinedAt"] = pd.to_datetime(
            users_df["joinedAt"], errors="coerce"
        ).dt.tz_localize(None)

        friends_df = pd.read_csv(
            users, dtype={"user_id": int}, parse_dates=["joinedAt"]
        )
        friends_df["birthyear"] = pd.to_numeric(
            friends_df["birthyear"], errors="coerce"
        )
        friends_df["joinedAt"] = pd.to_datetime(
            friends_df["joinedAt"], errors="coerce"
        ).dt.tz_localize(None)
        events_df = pd.read_csv(events)
        events_df["start_time"] = pd.to_datetime(
            events_df["start_time"], errors="coerce"
        ).dt.tz_localize(None)

        train = os.path.join(path, "train.csv")
        event_interest_df = pd.read_csv(train)
        event_interest_df["timestamp"] = pd.to_datetime(
            event_interest_df["timestamp"]
        ).dt.tz_localize(None)

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
                    df=event_attendees_flattened_df,
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
                    df=user_friends_flattened_df,
                    fkey_col_to_pkey_table={
                        "user": "users",
                        "friend": "friends",
                    },
                ),
            }
        )

        db = db.from_(pd.Timestamp("2012-06-20"))

        return db

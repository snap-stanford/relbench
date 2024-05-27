import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.event_recommendation import RecommendationTask
from relbench.tasks.f1 import DidNotFinishTask, PositionTask, QualifyingTask
from relbench.utils import decompress_gz_file


class EventRecommendationDataset(RelBenchDataset):
    name = "rel-event"
    url = "https://www.kaggle.com/competitions/event-recommendation-engine-challenge"  # noqa
    err_msg = (
        "{data} not found. Please download "
        "event-recommendation-engine-challenge.zip from "
        "'{url}' and move it to '{path}'. Once you have your"
        "Kaggle API key, you can use the following command: "
        "kaggle competitions download -c event-recommendation-engine-challenge"
    )

    train_start_timestamp = pd.Timestamp("2012-04-27 21:41:02.227000+00:00")
    val_timestamp = pd.Timestamp("2012-11-01")
    test_timestamp = pd.Timestamp("2012-11-11")
    max_eval_time_frames = 1
    task_cls_list = [RecommendationTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def check_table_and_decompress_if_exists(
        self, table_path: str, alt_path: str | None = None
    ):
        if not os.path.exists(table_path) or (
            alt_path is not None and not os.path.exists(alt_path)
        ):
            if os.path.exists(table_path + ".gz"):
                decompress_gz_file(table_path + ".gz", table_path)
            else:
                self.err_msg.format(data=table_path, url=self.url, path=table_path)

    def make_db(self) -> Database:
        path = os.path.join("data", "event-recommendation")
        zip = os.path.join(path, "event-recommendation-engine-challenge.zip")
        users = os.path.join(path, "users.csv")
        user_friends = os.path.join(path, "user_friends.csv")
        events = os.path.join(path, "events.csv")
        event_attendees = os.path.join(path, "event_attendees.csv")
        train = os.path.join(path, "train.csv")
        if not (os.path.exists(users)):
            if not os.path.exists(zip):
                raise RuntimeError(
                    self.err_msg.format(data="Dataset", url=self.url, path=zip)
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)
        self.check_table_and_decompress_if_exists(
            user_friends, os.path.join(path, "user_friends_flattened.csv")
        )
        self.check_table_and_decompress_if_exists(events)
        self.check_table_and_decompress_if_exists(
            event_attendees, os.path.join(path, "event_attendees_flattened.csv")
        )
        users_df = pd.read_csv(users, dtype={"user_id": int}, parse_dates=["joinedAt"])
        users_df["joinedAt"] = pd.to_datetime(users_df["joinedAt"], errors="coerce")
        friends_df = pd.read_csv(
            users, dtype={"user_id": int}, parse_dates=["joinedAt"]
        )
        friends_df["joinedAt"] = pd.to_datetime(friends_df["joinedAt"], errors="coerce")
        events_df = pd.read_csv(events)
        events_df["start_time"] = pd.to_datetime(
            events_df["start_time"], errors="coerce"
        )
        train_df = pd.read_csv(train)
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")

        if not os.path.exists(os.path.join(path, "user_friends_flattened.csv")):
            user_friends_df = pd.read_csv(user_friends)
            user_friends_df = (
                user_friends_df.set_index("user")["friends"]
                .str.split(expand=True)
                .stack()
                .reset_index()
            )
            user_friends_df.columns = ["user", "index", "friend"]
            user_friends_flattened_df = user_friends_df.drop("index", axis=1)
            user_friends_flattened_df["user"] = user_friends_flattened_df[
                "user"
            ].astype(int)
            user_friends_flattened_df["friend"] = user_friends_flattened_df[
                "friend"
            ].astype(int)
            user_friends_flattened_df.to_csv(
                os.path.join(path, "user_friends_flattened.csv")
            )
        else:
            user_friends_flattened_df = pd.read_csv(
                os.path.join(path, "user_friends_flattened.csv")
            )

        if not os.path.exists(os.path.join(path, "event_attendees_flattened.csv")):
            event_attendees_df = pd.read_csv(event_attendees)
            melted_df = event_attendees_df.melt(
                id_vars=["event"],
                value_vars=["yes", "maybe", "invited", "no"],
                var_name="status",
                value_name="user_ids",
            )
            melted_df = melted_df.dropna()
            melted_df["user_ids"] = melted_df["user_ids"].str.split()
            melted_df["user_ids"] = melted_df["user_ids"].apply(
                lambda x: [int(i) for i in x]
            )
            exploded_df = melted_df.explode("user_ids")
            exploded_df["user_ids"] = exploded_df["user_ids"].astype(int)
            exploded_df.rename(columns={"user_ids": "user_id"}, inplace=True)
            exploded_df = pd.merge(
                exploded_df,
                events_df[["event_id", "start_time"]],
                left_on="event",
                right_on="event_id",
                how="left",
            )
            exploded_df = exploded_df.drop("event_id", axis=1)
            event_attendees_flattened_df = exploded_df
            event_attendees_flattened_df.to_csv(
                os.path.join(path, "event_attendees_flattened.csv")
            )
        else:
            event_attendees_flattened_df = pd.read_csv(
                os.path.join(path, "event_attendees_flattened.csv")
            )
            event_attendees_flattened_df["start_time"] = pd.to_datetime(
                event_attendees_flattened_df["start_time"], errors="coerce"
            )

        import pdb

        pdb.set_trace()
        return Database(
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
                "user_friends": Table(
                    df=user_friends_flattened_df,
                    fkey_col_to_pkey_table={
                        "user": "users",
                        "friend": "friends",
                    },
                ),
                "train": Table(
                    df=train_df,
                    fkey_col_to_pkey_table={"user": "users", "event": "events"},
                    time_col="timestamp",
                ),
            }
        )

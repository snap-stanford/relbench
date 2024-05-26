import os

import numpy as np
import pandas as pd
import pooch
import shutil
from pathlib import Path

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.f1 import DidNotFinishTask, PositionTask, QualifyingTask
from relbench.datasets import decompress_gz_file


class EventRecommendationDataset(RelBenchDataset):
    name = "rel-event"
    url = "https://www.kaggle.com/competitions/event-recommendation-engine-challenge" # noqa
    err_msg = ("{data} not found. Please download "
                "event-recommendation-engine-challenge.zip from "
                "'{url}' and move it to '{path}'. Once you have your"
                "Kaggle API key, you can use the following command: "
                "kaggle competitions download -c event-recommendation-engine-challenge"
                )

    train_start_timestamp = pd.Timestamp('2012-04-27 21:41:02.227000+00:00')
    val_start_timestamp = pd.Timestamp('2012-11-01')
    test_start_timestamp = pd.Timestamp('2012-11-11')

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def check_table_and_decompress_if_exists(self, table_path: str):
        if not os.path.exists(table_path):
            if os.path.exists(table_path + '.gz'):
                decompress_gz_file(table_path + '.gz', table_path)
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
                    self.err_msg.format(data='Dataset', url=self.url, path=zip)
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)
        self.check_table_and_decompress_if_exists(user_friends)
        self.check_table_and_decompress_if_exists(events)
        self.check_table_and_decompress_if_exists(event_attendees)
        (user_friends.set_index('user')['friends'].str.split(expand=True).stack().reset_index())
        user_friends.columns = ['user', 'index', 'friend']
        user_friends_flattened_df = user_friends.drop('index', axis=1)
        melted_df = event_attendees.melt(id_vars=['event'], value_vars=['yes', 'maybe', 'invited', 'no'], var_name='status', value_name='user_ids')
        melted_df = melted_df.dropna()
        melted_df['user_ids'] = melted_df['user_ids'].str.split()
        exploded_df = melted_df.explode('user_ids')
        exploded_df.rename(columns={'user_ids': 'user'}, inplace=True)
        event_attendees_flattened_df = exploded_df
        events_df = pd.read_csv(events)
        users_df = pd.read_csv(users)
        friends_df = pd.read_csv(users)
        train_df = pd.read_csv(train)
        return Database(
            table_dict={
                'users': Table(
                    df=users_df,
                    fkey_col_to_pkey_table={},
                    pkey_col='user_id',
                    time_col='joinedAt',
                ),
                "friends": Table(
                    df=friends_df,
                    fkey_col_to_pkey_table={},
                    pkey_col='user_id',
                    time_col='joinedAt'
                ),
                "events": Table(
                    df=events_df,
                    fkey_col_to_pkey_table={},
                    pkey_col='event_id',
                    time_col='start_time',
                ),
                "event_attendees": Table(
                    df=event_attendees_flattened_df,
                    fkey_col_to_pkey_table={
                        "event": "events",
                        "user": "users",
                    }
                ),
                "user_friends": Table(
                    df=user_friends_flattened_df,
                    fkey_col_to_pkey_table={
                        "user": "users",
                        "friend": "friends",
                    }
                ),
                "train": Table(
                    df=train_df,
                    fkey_col_to_pkey_table={
                        "user": "users",
                        "event": "events"
                    },
                    time_col="timestamp"
                )
            }
        )

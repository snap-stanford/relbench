import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchTask, Table
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window


class EngageTask(RelBenchTask):
    r"""Predict if a user will make any votes/posts/comments in the next 3 years."""

    name = "rel-stackex-engage"
    task_type = "binary_classification"
    entity_col = "OwnerUserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "contribution"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserContributionTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        comments = db.table_dict["comments"].df
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df
        posts = posts[
            posts.OwnerUserId != -1
        ]  ## when user id is -1, it is stats exchange community, not a real person
        posts = posts[posts.OwnerUserId.notnull()]  ## 1153 null posts

        users = db.table_dict["users"].df
        votes = votes[votes.UserId.notnull()]
        posts = posts[posts.OwnerUserId.notnull()]

        comments = comments[
            comments.UserId != -1
        ]  ## when user id is -1, it is stats exchange community, not a real person
        comments = comments[comments.UserId.notnull()]  ## 2439 null comments

        def get_values_in_window(row, posts, users):
            posts_window = get_df_in_window(posts, "CreationDate", row, self.timedelta)
            comments_window = get_df_in_window(
                comments, "CreationDate", row, self.timedelta
            )
            votes_window = get_df_in_window(votes, "CreationDate", row, self.timedelta)

            user_made_posts_in_this_period = posts_window.OwnerUserId.unique()
            user_made_comments_in_this_period = comments_window.UserId.unique()
            user_made_votes_in_this_period = votes_window.UserId.unique()

            # user_active_in_this_period = user_made_posts_in_this_period

            user_active_in_this_period = np.union1d(
                np.union1d(
                    user_made_posts_in_this_period, user_made_comments_in_this_period
                ),
                user_made_votes_in_this_period,
            )

            users_exist = users[
                users.CreationDate <= row["timestamp"]
            ]  ## only looking at existing users
            users_exist_ids = users_exist.Id.values

            user_made_votes = votes[
                votes.CreationDate <= row["timestamp"]
            ].UserId.unique()
            user_made_comments = comments[
                comments.CreationDate <= row["timestamp"]
            ].UserId.unique()
            user_made_posts = posts[
                posts.CreationDate <= row["timestamp"]
            ].OwnerUserId.unique()
            active_user_before_this_time = np.union1d(
                np.union1d(user_made_votes, user_made_comments), user_made_posts
            )
            # active_user_before_this_time = user_made_posts
            users_exist_and_active_ids = np.intersect1d(
                users_exist_ids, active_user_before_this_time
            )

            user2churn = pd.DataFrame()
            user2churn["OwnerUserId"] = users_exist_and_active_ids
            user2churn["timestamp"] = row["timestamp"]

            user2churn["contribution"] = user2churn.OwnerUserId.apply(
                lambda x: 1 if x in user_active_in_this_period else 0
            )  ## 1: contributed; 0: not contributed
            return user2churn

        tqdm.pandas()
        # Apply function to each time window
        res = timestamp_df.progress_apply(
            lambda row: get_values_in_window(row, posts, users), axis=1
        )
        df = pd.concat(res.values)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class VotesTask(RelBenchTask):
    r"""Predict the number of upvotes that a question that is posted within the last 2 years will receive in the next 6 months. ?"""
    name = "rel-stackex-votes"
    task_type = "regression"
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=180)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for post_votes_next_month."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        votes = db.table_dict["votes"].df
        votes = votes[votes.PostId.notnull()]
        votes = votes[votes.VoteTypeId == 2]  ## upvotes

        posts = db.table_dict["posts"].df
        posts = posts[
            posts.OwnerUserId != -1
        ]  ## when user id is -1, it is stats exchange community, not a real person
        posts = posts[posts.OwnerUserId.notnull()]  ## 1153 null posts

        posts = posts[posts.PostTypeId == 1]  ## just looking at questions

        def get_values_in_window(row, votes, posts):
            votes_window = get_df_in_window(votes, "CreationDate", row, self.timedelta)
            posts_exist = posts[
                (posts.CreationDate <= row["timestamp"])
                & (posts.CreationDate > (row["timestamp"] - pd.Timedelta(days=365 * 2)))
            ]  ## posts exist and active defined by created in the last 2 years
            posts_exist_ids = posts_exist.Id.values
            train_table = pd.DataFrame()
            train_table["PostId"] = posts_exist_ids
            train_table["timestamp"] = row["timestamp"]

            num_of_upvotes = dict(votes_window.groupby("PostId")["Id"].agg(len))
            train_table["popularity"] = train_table.PostId.apply(
                lambda x: num_of_upvotes[x] if x in num_of_upvotes else 0
            )  ## default all existing users have 0 comment scores
            return train_table

        tqdm.pandas()
        # Apply function to each row in df_b
        res = timestamp_df.progress_apply(
            lambda row: get_values_in_window(row, votes, posts), axis=1
        )
        df = pd.concat(res.values)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

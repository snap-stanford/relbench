import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchNodeTask, RelBenchLinkTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, hits_at_k, mrr
from relbench.utils import get_df_in_window


class EngageTask(RelBenchNodeTask):
    r"""Predict if a user will make any votes/posts/comments in the next 1 year."""

    name = "rel-stackex-engage"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "OwnerUserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "contribution"
    timedelta = pd.Timedelta(days=365)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserContributionTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        comments = db.table_dict["comments"].df
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df
        users = db.table_dict["users"].df

        df = duckdb.sql(
            f"""
            WITH
            ALL_ENGAGEMENT AS (
                SELECT
                    p.id,
                    p.owneruserid as userid,
                    p.creationdate
                FROM posts p
                UNION
                SELECT
                    v.id,
                    v.userid,
                    v.creationdate
                FROM votes v
                UNION
                SELECT
                    c.id,
                    c.userid,
                    c.creationdate
                FROM comments c
            ),

            ACTIVE_USERS AS (
                 SELECT
                    t.timestamp,
                    u.id,
                    count(distinct a.id) as n_engagement
                FROM timestamp_df t
                CROSS JOIN users u
                LEFT JOIN all_engagement a
                ON u.id = a.UserId
                    and a.CreationDate <= t.timestamp
                WHERE u.id != -1
                GROUP BY t.timestamp, u.id
             )
                SELECT
                    u.timestamp,
                    u.id as OwnerUserId,
                    IF(count(distinct a.id) >= 1, 1, 0) as contribution
                FROM active_users u
                LEFT JOIN all_engagement a
                ON u.id = a.UserId
                    and a.CreationDate > u.timestamp
                    and a.CreationDate <= u.timestamp + INTERVAL '{self.timedelta}'
                where u.n_engagement >= 1
                GROUP BY u.timestamp, u.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class VotesTask(RelBenchNodeTask):
    r"""Predict the number of upvotes that a question that is posted within the
    last 1 year will receive in the next 1 year."""
    name = "rel-stackex-votes"
    task_type = TaskType.REGRESSION
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=365)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for post_votes_next_month."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    p.id as PostId,
                    count(distinct v.id) as popularity
                FROM timestamp_df t
                LEFT JOIN posts p
                ON p.CreationDate > t.timestamp - INTERVAL '{self.timedelta} days'
                and p.CreationDate <= t.timestamp
                and p.owneruserid != -1
                and p.owneruserid is not null
                and p.PostTypeId = 1
                LEFT JOIN votes v
                ON p.id = v.PostId
                and v.CreationDate > t.timestamp
                and v.CreationDate <= t.timestamp + INTERVAL '{self.timedelta}'
                and v.votetypeid = 2
                GROUP BY t.timestamp, p.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )



class UserCommentOnPostTask(RelBenchLinkTask):
    r"""Predict if a user will comment on a specific post within 24hrs of the post being made."""

    name = "rel-stackex-comment-on-post"
    task_type = TaskType.BINARY_CLASSIFICATION
    source_entity_col = "UserId"
    source_entity_table = "users"
    destination_entity_col = "PostId"
    destination_entity_table = "posts"
    time_col = "timestamp"
    target_col = "target"
    timedelta = pd.Timedelta(days=365)
    metrics = [(hits_at_k, 10), (hits_at_k, 20), (hits_at_k, 30), (mrr, None)]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserCommentOnPostTask."""
        
        timestamp_df = pd.DataFrame({"timestamp": timestamps})    

        users = db.table_dict["users"].df
        posts = db.table_dict["posts"].df 
        comments = db.table_dict["comments"].df

        df = duckdb.sql(
                    f"""
                        SELECT
                            t.timestamp,
                            p.id as PostId,
                            c.UserId as UserId
                        FROM timestamp_df t
                        LEFT JOIN posts p
                        ON p.CreationDate > t.timestamp - INTERVAL '{2 * self.timedelta} days'
                        and p.CreationDate <= t.timestamp
                        LEFT JOIN comments c
                        ON p.id = c.PostId
                        and c.CreationDate > t.timestamp
                        and c.CreationDate <= t.timestamp + INTERVAL '{self.timedelta} days'
                        where c.UserId is not null and p.owneruserid != -1 and p.owneruserid is not null
                    ;
                    """
                ).df()
        

        # add 'target' column of all 1s
        df[self.target_col] = np.ones(len(df))


        ########### Negative Link Sampling 

        # TODO (joshrob) check for false negatives
        # TODO (joshrob) save negative links to disk to avoid resampling

        NUM_NEGATIVES = 1000

        # randomly sample NUM_NEGATIVE negative pairs   
        users_arr = users[db.table_dict["users"].pkey_col].to_numpy()
        timestamp_arr = posts[db.table_dict["posts"].time_col].to_numpy()
        posts_arr = posts[db.table_dict["posts"].pkey_col].to_numpy()

        perm_users = np.random.permutation(len(users))[:NUM_NEGATIVES]
        neg_UserIDs = users_arr[perm_users]

        perm_posts = np.random.permutation(len(posts))[:NUM_NEGATIVES]
        neg_PostIDs = posts_arr[perm_posts]

        timestamp_arr = timestamp_arr[perm_posts]

        # create dataframe with negative pairs 

        df_neg = pd.DataFrame({self.source_entity_col: neg_UserIDs, # WARNING: this is not the same as self.source_entity_col
                               self.destination_entity_col: neg_PostIDs, # WARNING: this is not the same as self.destination_entity_col
                               self.time_col: timestamp_arr,
                               self.target_col: np.zeros(len(neg_UserIDs))
                               })


        df = pd.concat([df, df_neg], ignore_index=True)


        return Table(
            df=df,
            fkey_col_to_pkey_table={self.source_entity_col: self.source_entity_table,
                                   self.destination_entity_col: self.destination_entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
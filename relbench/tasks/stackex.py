import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchNodeTask, RelBenchLinkTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, hits_at_k_or_mrr
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
    source_entity_col = "Id"
    source_entity_table = "users"
    destination_entity_col = "Id"
    destination_entity_table = "posts"
    time_col = "timestamp"
    target_col = "target"
    timedelta = pd.Timedelta(days=1)
    metrics = [(hits_at_k_or_mrr, k, mrr) for k, mrr in [(10, False), (20, False), (30, False), (-1, True)]]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserCommentOnPostTask."""
        
        df = pd.DataFrame(columns=[self.source_entity_col, self.destination_entity_col, 'timestamp', self.target_col])

        
        # step 1: 

        ########### Positive Link Sampling

        #     - make table with all (UserID, PostID) pairs for which UserID does comment on PostID in 24hrs. 
        #     - timestamp = time post was made. 
        #     - target = 1 for all rows
        
        
        ########### Negative Link Sampling 

        # TODO (joshrob) check for false negatives
        # TODO (joshrob) save negative links to disk to avoid resampling

        users = db.table_dict["users"].df[self.source_entity_col].to_numpy()
        posts = db.table_dict["posts"].df[self.destination_entity_col].to_numpy()  
        post_timestamps = db.table_dict["posts"].df[db.table_dict["posts"].time_col].to_numpy()

        NUM_NEGATIVES = 1000


        # randomly sample NUM_NEGATIVE negative pairs   

        perm_users = np.random.permutation(len(users))[:NUM_NEGATIVES]
        neg_UserIDs = users[perm_users]

        perm_posts = np.random.permutation(len(posts))[:NUM_NEGATIVES]
        neg_PostIDs = posts[perm_posts]
        post_timestamps = post_timestamps[perm_posts]

        # create dataframe with negative pairs 

        df_neg = pd.DataFrame({"UserId": neg_UserIDs, # WARNING: this is not the same as self.source_entity_col
                               "PostId": neg_PostIDs, # WARNING: this is not the same as self.destination_entity_col
                               'timestamp': post_timestamps,
                               self.target_col: np.zeros(len(neg_UserIDs))
                               })


        #df = pd.concat([df, df_neg], ignore_index=True)
        df = df_neg

        return Table(
            df=df,
            fkey_col_to_pkey_table={"UserId": self.source_entity_table,
                                   "PostId": self.destination_entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
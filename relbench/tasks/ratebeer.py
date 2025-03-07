import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy, 
    average_precision, 
    f1, 
    link_prediction_map,
    link_prediction_precision, 
    link_prediction_recall, 
    mae, 
    r2, 
    rmse, 
    roc_auc
)


class UserBeerRatingTask(EntityTask):
    r"""Predict the average rating a user will give to beers in the next month.
    
    Looks at a user's past rating behavior to predict their average rating
    across all beers they will rate in the next 30 days.
    """

    task_type = TaskType.REGRESSION
    entity_col = "user_id"
    entity_table = "users"
    time_col = "date"
    target_col = "avg_rating"
    timedelta = pd.Timedelta(days=30)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        beer_ratings = db.table_dict["beer_ratings"].df
        users = db.table_dict["users"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                u.user_id,
                AVG(br.total_score) as avg_rating
            FROM
                timestamp_df t
            CROSS JOIN
                users u
            LEFT JOIN
                beer_ratings br
            ON
                br.user_id = u.user_id AND
                br.created_at > t.timestamp AND
                br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                u.user_id IN (
                    SELECT DISTINCT user_id
                    FROM beer_ratings
                    WHERE created_at > t.timestamp - INTERVAL '180 days'
                    AND created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp, u.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class BeerRatingTask(EntityTask):
    r"""Predict the average rating a beer will receive in the next month.
    
    Uses the beer's past ratings to predict the average score it will receive
    from all users who rate it in the next 30 days.
    """

    task_type = TaskType.REGRESSION
    entity_col = "beer_id"
    entity_table = "beers"
    time_col = "date"
    target_col = "avg_rating"
    timedelta = pd.Timedelta(days=30)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        beer_ratings = db.table_dict["beer_ratings"].df
        beers = db.table_dict["beers"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                b.beer_id,
                AVG(br.total_score) as avg_rating
            FROM
                timestamp_df t
            CROSS JOIN
                beers b
            LEFT JOIN
                beer_ratings br
            ON
                br.beer_id = b.beer_id AND
                br.created_at > t.timestamp AND
                br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                b.beer_id IN (
                    SELECT DISTINCT beer_id
                    FROM beer_ratings
                    WHERE created_at > t.timestamp - INTERVAL '180 days'
                    AND created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp, b.beer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class BeerPopularityTask(EntityTask):
    r"""Predict how many ratings a beer will receive in the next month.
    
    Estimates a beer's popularity by predicting the total number of ratings
    it will receive in the next 30 days.
    """

    task_type = TaskType.REGRESSION
    entity_col = "beer_id"
    entity_table = "beers"
    time_col = "date"
    target_col = "rating_count"
    timedelta = pd.Timedelta(days=30)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        beer_ratings = db.table_dict["beer_ratings"].df
        beers = db.table_dict["beers"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                b.beer_id,
                COUNT(br.rating_id) as rating_count
            FROM
                timestamp_df t
            CROSS JOIN
                beers b
            LEFT JOIN
                beer_ratings br
            ON
                br.beer_id = b.beer_id AND
                br.created_at > t.timestamp AND
                br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                b.beer_id IN (
                    SELECT DISTINCT beer_id
                    FROM beer_ratings
                    WHERE created_at > t.timestamp - INTERVAL '180 days'
                    AND created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp, b.beer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class UserBeerRecommendationTask(RecommendationTask):
    r"""Predict which beers a user will rate in the next month.
    
    Link prediction task that recommends beers to users by predicting which
    beers they will choose to rate in the next 30 days.
    """

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "date"
    timedelta = pd.Timedelta(days=30)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        beer_ratings = db.table_dict["beer_ratings"].df
        users = db.table_dict["users"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                br.user_id,
                LIST(DISTINCT br.beer_id) as beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings br
            ON
                br.created_at > t.timestamp AND
                br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                br.user_id IN (
                    SELECT DISTINCT user_id
                    FROM beer_ratings
                    WHERE created_at > t.timestamp - INTERVAL '180 days'
                    AND created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp, br.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserBeerScoreTask(EntityTask):
    r"""Predict the specific rating a user would give to a particular beer.
    
    This is a matrix completion task where we predict the exact rating score
    that a specific user would give to a specific beer if they were to rate it
    in the next 30 days.
    """
    
    task_type = TaskType.REGRESSION
    entity_col = ["user_id", "beer_id"]  # This task operates on user-beer pairs
    entity_table = ["users", "beers"]    # Both tables are needed
    time_col = "date"
    target_col = "rating_score"
    timedelta = pd.Timedelta(days=30)
    metrics = [r2, mae, rmse]
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        beer_ratings = db.table_dict["beer_ratings"].df
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        
        # First, find active users and beers during the prediction period
        df = duckdb.sql(
            f"""
            WITH active_users AS (
                SELECT DISTINCT user_id
                FROM beer_ratings
                WHERE created_at > timestamp_df.timestamp - INTERVAL '180 days'
                  AND created_at <= timestamp_df.timestamp
            ),
            active_beers AS (
                SELECT DISTINCT beer_id
                FROM beer_ratings
                WHERE created_at > timestamp_df.timestamp - INTERVAL '180 days'
                  AND created_at <= timestamp_df.timestamp
            ),
            -- Find ratings that occurred in the target period
            future_ratings AS (
                SELECT 
                    user_id, 
                    beer_id, 
                    total_score as rating_score
                FROM beer_ratings
                WHERE created_at > timestamp_df.timestamp 
                  AND created_at <= timestamp_df.timestamp + INTERVAL '{self.timedelta} days'
            )
            -- Select sample pairs that actually were rated in the future 
            SELECT
                timestamp_df.timestamp as date,
                fr.user_id,
                fr.beer_id,
                fr.rating_score
            FROM
                timestamp_df
            CROSS JOIN
                future_ratings fr
            WHERE
                fr.user_id IN (SELECT user_id FROM active_users) AND
                fr.beer_id IN (SELECT beer_id FROM active_beers)
            """
        ).df()
        
        # Create a table with multiple entity columns
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                "user_id": "users",
                "beer_id": "beers"
            },
            pkey_col=None,
            time_col=self.time_col,
        )

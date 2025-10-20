import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata

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
    roc_auc,
)


def link_prediction_mrr(
    pred_isin: NDArray[np.int_],  # shape (n_src, k)
    dst_count: NDArray[np.int_],  # shape (n_src,)
) -> float:
    # 1) filter out sources without positives
    pos_mask = dst_count > 0
    pred_isin = pred_isin[pos_mask]
    dst_count = dst_count[pos_mask]

    reciprocal_ranks = []
    for row in pred_isin:
        # find the first correct prediction
        hits = np.where(row == 1)[0]
        if len(hits):
            first_hit_rank = hits[0] + 1  # ranks are 1‑based
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


# Entity Classification tasks


class BeerRatingChurnTask(EntityTask):
    r"""Predict whether a beer will receive a rating in the next 90 days."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "beer_id"
    entity_table = "beers"
    time_col = "timestamp"
    target_col = "rating_churn"
    timedelta = pd.Timedelta(days=90)
    metrics = [accuracy, average_precision, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        beers = db.table_dict["beers"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                b.beer_id AS beer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM beer_ratings
                        WHERE
                            beer_ratings.beer_id = b.beer_id
                            AND beer_ratings.created_at > t.timestamp
                            AND beer_ratings.created_at <= t.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS rating_churn
            FROM
                timestamp_df t,
                beers b
            WHERE
                EXISTS (
                    SELECT 1
                    FROM beer_ratings AS br2
                    WHERE
                        br2.beer_id = b.beer_id
                        AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta}'
                        AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                b.beer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class UserRatingChurnTask(EntityTask):
    r"""Predict whether a user will give a beer rating in the next 90 days."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "user_id"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "user_churn"
    timedelta = pd.Timedelta(days=90)
    metrics = [accuracy, average_precision, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                u.user_id AS user_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM beer_ratings
                        WHERE
                            beer_ratings.user_id = u.user_id
                            AND beer_ratings.created_at > t.timestamp
                            AND beer_ratings.created_at <= t.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS user_churn
            FROM
                timestamp_df t,
                users u
            WHERE
                EXISTS (
                    SELECT 1
                    FROM beer_ratings AS br2
                    WHERE
                        br2.user_id = u.user_id
                        AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta}'
                        AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                u.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class BrewerDormantTask(EntityTask):
    r"""Predict whether a brewer will release zero beers in the next 365 days."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "brewer_id"
    entity_table = "brewers"
    time_col = "timestamp"
    target_col = "dormant"
    timedelta = pd.Timedelta(days=365)
    metrics = [accuracy, average_precision, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        brewers = db.table_dict["brewers"].df
        beers = db.table_dict["beers"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                brew.brewer_id AS brewer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM beers
                        WHERE
                            beers.brewer_id = brew.brewer_id
                            AND beers.created_at > t.timestamp
                            AND beers.created_at <= t.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS dormant
            FROM
                timestamp_df t,
                brewers brew
            WHERE
                EXISTS (
                    SELECT 1
                    FROM beers AS b2
                    WHERE
                        b2.brewer_id = brew.brewer_id
                        AND b2.created_at > t.timestamp - INTERVAL '{self.timedelta}'
                        AND b2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                brew.brewer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


# Entity Regression tasks


class UserRatingCountTask(EntityTask):
    r"""Predict the number of beer ratings that a user will give in the next 90 days."""

    task_type = TaskType.REGRESSION
    entity_col = "user_id"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "num_ratings"
    timedelta = pd.Timedelta(days=90)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                u.user_id,
                (
                    SELECT COUNT(*)
                    FROM beer_ratings br
                    WHERE br.user_id = u.user_id
                    AND br.created_at >  t.timestamp
                    AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta}'
                ) AS num_ratings
            FROM timestamp_df t
            CROSS JOIN users u
            WHERE
                EXISTS (
                    SELECT 1
                    FROM beer_ratings AS br2
                    WHERE br2.user_id = u.user_id
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta}'
                    AND br2.created_at <= t.timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


# Recommendation tasks


class UserFavoriteBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each active user will add to their favorites
    in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [
        link_prediction_precision,
        link_prediction_recall,
        link_prediction_map,
        link_prediction_mrr,
    ]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        favorites = db.table_dict["favorites"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                f.user_id,
                LIST(DISTINCT f.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                favorites as f
            ON
                f.created_at > t.timestamp AND
                f.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                f.user_id is not null and f.beer_id is not null
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br
                    WHERE br.user_id = f.user_id
                    AND br.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                f.user_id
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


class UserLikedPlaceTask(RecommendationTask):
    r"""Predict the list of distinct places each active user rates at least 80.0 / 100.0
    in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "place_id"
    dst_entity_table = "places"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [
        link_prediction_precision,
        link_prediction_recall,
        link_prediction_map,
        link_prediction_mrr,
    ]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        places = db.table_dict["places"].df
        place_ratings = db.table_dict["place_ratings"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                pr.user_id,
                LIST(DISTINCT pr.place_id) AS place_id
            FROM
                timestamp_df t
            LEFT JOIN
                place_ratings as pr
                ON pr.created_at > t.timestamp
                AND pr.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                pr.user_id IS NOT NULL and pr.place_id IS NOT NULL
                AND pr.total_score >= 80
                AND EXISTS (
                    SELECT 1
                    FROM place_ratings as pr2
                    WHERE pr2.user_id = pr.user_id
                    AND pr2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND pr2.created_at <= t.timestamp
                )
                OR EXISTS (
                    SELECT 1
                    FROM beer_ratings as br
                    WHERE br.user_id = pr.user_id
                    AND br.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                pr.user_id
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


class UserLikedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each active user rates at least 4.0 / 5.0 in
    the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [
        link_prediction_precision,
        link_prediction_recall,
        link_prediction_map,
        link_prediction_mrr,
    ]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                br.user_id,
                LIST(DISTINCT br.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings as br
                ON br.created_at > t.timestamp
                AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                br.user_id IS NOT NULL and br.beer_id IS NOT NULL
                AND br.total_score >= 4.0
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                br.user_id
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


# Mapping of task names to their corresponding task classes.
tasks_dict = {
    "beer-rating-churn": BeerRatingChurnTask,
    "user-rating-churn": UserRatingChurnTask,
    "brewer-dormant": BrewerDormantTask,
    "user-rating-count": UserRatingCountTask,
    "user-favorite-beer": UserFavoriteBeerTask,
    "user-liked-place": UserLikedPlaceTask,
    "user-liked-beer": UserLikedBeerTask,
}

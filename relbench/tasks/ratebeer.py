import duckdb
import pandas as pd
from relbench.base import Database, RecommendationTask, Table, TaskType, EntityTask
from relbench.metrics import (
    link_prediction_precision,
    link_prediction_recall,
    link_prediction_map,
    roc_auc, accuracy, f1, average_precision
)

class UserBeerPreferenceTask(RecommendationTask):
    """
    NB‑HRP – Predict beers a user will rate >=3.8 in the next 180 days.
    """

    # ---------- static metadata ----------
    task_type = TaskType.LINK_PREDICTION
    src_entity_col, src_entity_table = "user_id", "users"
    dst_entity_col, dst_entity_table = "beer_id", "beers"
    time_col = "timestamp"

    # --------------- config ---------------
    timedelta = pd.Timedelta(days=180)             # prediction horizon
    eval_k    = 10
    metrics   = [
        link_prediction_precision,
        link_prediction_recall,
        link_prediction_map
    ]

    # ------------- table builder -------------
    def make_table(
        self,
        db: Database,
        timestamps: "pd.Series[pd.Timestamp]",
    ) -> Table:
        """Return a Table with columns [timestamp, user_id, beer_id (LIST)]."""
        # ---- 0. sanity on input timestamps ----
        if timestamps.empty:
            raise ValueError("`timestamps` Series is empty.")

        ts_df = pd.DataFrame({self.time_col: timestamps})

        # ---- 1. pull and prepare ratings ----
        ratings = db.table_dict["beer_ratings"].df.copy()

        # Accept either 'created_at' or 'rating_time'
        if "created_at" in ratings.columns and "rating_time" not in ratings.columns:
            ratings = ratings.rename(columns={"created_at": "rating_time"})
        if "rating_time" not in ratings.columns:
            raise KeyError("beer_ratings must contain 'created_at' or 'rating_time'")

        ratings["rating_time"] = pd.to_datetime(ratings["rating_time"])
        ratings["good"] = (ratings["total_score"] >= 3.8).astype("int8")

        # ---- 2. DuckDB query ----
        days = int(self.timedelta.total_seconds() // 86_400)  # 180 for interval
        con  = duckdb.connect()
        con.register("ts", ts_df)
        con.register("ratings", ratings)

        sql = f"""
        WITH active AS (
            -- users active in the 365 days *before* each timestamp
            SELECT
                ts.{self.time_col},
                r.user_id
            FROM ts
            JOIN ratings r
              ON r.rating_time > ts.{self.time_col} - INTERVAL 365 DAY
             AND r.rating_time <= ts.{self.time_col}
            GROUP BY 1, 2
        ),
        pos AS (
            -- beers first rated >=3.8 in (t, t+Δ]
            SELECT
                ts.{self.time_col},
                r.user_id,
                LIST(DISTINCT r.beer_id) AS {self.dst_entity_col}
            FROM ts
            JOIN ratings r
              ON r.good = 1
             AND r.rating_time >  ts.{self.time_col}
             AND r.rating_time <= ts.{self.time_col} + INTERVAL {days} DAY
            JOIN active a
              ON a.{self.time_col} = ts.{self.time_col}
             AND a.user_id       = r.user_id
            GROUP BY 1, 2
        )
        SELECT * FROM pos
        ORDER BY {self.time_col}, user_id;
        """
        df = con.execute(sql).fetchdf()
        con.close()

        # ---- 3. wrap into RelBench Table ----
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )



class BeerStylePredictionTask(EntityTask):
    """
    BS‑TPC – Predict the style (style_id) of a newly added beer.

    - Task type: Multiclass classification
    - Entity: beer_id
    - Label: style_id
    - Anchor time: beers.created_at
    """

    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "beer_id"
    entity_table = "beers"
    time_col = "timestamp"
    target_col = "style_id"
    timedelta = pd.Timedelta(days=180)
    metrics = [accuracy, f1, average_precision, roc_auc]
    num_eval_timestamps = 1
    
    def _get_table(self, split: str) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=False)

        if split == "train":
            start, end = pd.Timestamp.min, self.dataset.val_timestamp
        elif split == "val":
            start, end = self.dataset.val_timestamp, self.dataset.test_timestamp
        elif split == "test":
            start, end = self.dataset.test_timestamp, db.max_timestamp
        else:
            raise ValueError(f"Unknown split: {split}")

        beers = db.table_dict["beers"].df.copy()
        beers = beers.dropna(subset=["style_id"])
        beers = beers[beers["style_id"] != ""]
        beers["created_at"] = pd.to_datetime(beers["created_at"])

        beers = beers[(beers["created_at"] >= start) & (beers["created_at"] < end)]
        beers["timestamp"] = beers["created_at"]

        return Table(
            df=beers[["timestamp", "beer_id", "style_id"]],
            fkey_col_to_pkey_table={"beer_id": "beers"},
            pkey_col="beer_id",
            time_col="timestamp",
        )


    def make_table(self, db: Database, timestamps: pd.Series) -> Table:
        if len(timestamps) == 0:
            raise ValueError("No timestamps provided.")

        beers = db.table_dict["beers"].df.copy()
        beers["created_at"] = pd.to_datetime(beers["created_at"])
        beers = beers.dropna(subset=["style_id"])
        beers = beers[beers["style_id"] != ""]
        beers = beers[beers["style_id"].notna()]
        beers["style_id"] = beers["style_id"].astype(int) 

        ts_sorted = pd.Series(sorted(timestamps)) 

        beers["timestamp"] = beers["created_at"].apply(
            lambda t: ts_sorted[ts_sorted <= t].max() if (ts_sorted <= t).any() else pd.NaT
        )
        beers = beers.dropna(subset=["timestamp"])

        return Table(
            df=beers[["timestamp", "beer_id", "style_id"]],
            fkey_col_to_pkey_table={"beer_id": "beers"},
            pkey_col="beer_id",
            time_col="timestamp",
        )






class BeerAvailabilityForecastTask(EntityTask):
    """
    BAF – Predict if a beer will be available at a place in the next 180 days.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "row_id"  # synthetic row ID
    entity_table = "availability"  # not used directly
    time_col = "timestamp"
    target_col = "label"
    timedelta = pd.Timedelta(days=180)
    metrics = [accuracy, f1, average_precision, roc_auc]
    num_eval_timestamps = 1

    def make_table(
        self,
        db: Database,
        timestamps: pd.Series,
    ) -> Table:
        if timestamps.empty:
            raise ValueError("No timestamps provided.")

        ts_df = pd.DataFrame({self.time_col: timestamps})
        availability = db.table_dict["availability"].df.copy()
        availability["created_at"] = pd.to_datetime(availability["created_at"])

        con = duckdb.connect()
        con.register("ts", ts_df)
        con.register("availability", availability)

        days = int(self.timedelta.total_seconds() // 86400)

        sql = f"""
        -- 1. Find true (beer, place) pairs in future window
        WITH future_pos AS (
            SELECT
                ts.{self.time_col},
                a.beer_id,
                a.place_id,
                1 AS label
            FROM ts
            JOIN availability a
              ON a.created_at > ts.{self.time_col}
             AND a.created_at <= ts.{self.time_col} + INTERVAL {days} DAY
        ),

        -- 2. Sample negative (beer, place) pairs
        all_beers AS (SELECT DISTINCT beer_id FROM availability),
        all_places AS (SELECT DISTINCT place_id FROM availability),
        all_pairs AS (
            SELECT
                ts.{self.time_col},
                b.beer_id,
                p.place_id
            FROM ts
            CROSS JOIN all_beers b
            CROSS JOIN all_places p
        ),
        negatives AS (
            SELECT
                ap.{self.time_col},
                ap.beer_id,
                ap.place_id,
                0 AS label
            FROM all_pairs ap
            LEFT JOIN future_pos fp
              ON fp.{self.time_col} = ap.{self.time_col}
             AND fp.beer_id = ap.beer_id
             AND fp.place_id = ap.place_id
            WHERE fp.beer_id IS NULL
            USING SAMPLE 1 PERCENT  -- You can also LIMIT based on positives count
        ),

        combined AS (
            SELECT * FROM future_pos
            UNION ALL
            SELECT * FROM negatives
        )
        SELECT
            ROW_NUMBER() OVER () AS row_id,
            *
        FROM combined
        ORDER BY {self.time_col}, beer_id, place_id;
        """

        df = con.execute(sql).fetchdf()
        con.close()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"beer_id": "beers", "place_id": "places"},
            pkey_col=self.entity_col,
            time_col=self.time_col,
        )


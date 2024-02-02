import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchTask, Table
from relbench.data.task import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window


class PointsTask(RelBenchTask):
    r"""Predict the finishing position of each driver in a race."""
    name = "rel-f1-points"
    task_type = TaskType.REGRESSION  # TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "points"
    timedelta = pd.Timedelta(days=365)
    metrics = [mae, rmse]  # [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for results_position_next_race."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    sum(re.points) points
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        # make into binary classification task
        # df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 0. else 0)

        # mean = 3.021777777777778
        # std = 5.299464335614526
        # df[self.target_col] = (df[self.target_col] - mean) / std

        # df[self.target_col] = df[self.target_col].apply(lambda x: np.log(x+1))

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if "val" not in self._cached_table_dict:
            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    self.dataset.val_timestamp,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["val"] = table
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if "full_test" not in self._cached_table_dict:
            full_table = self.make_table(
                self.dataset._full_db,
                pd.date_range(
                    self.dataset.test_timestamp,
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["full_test"] = full_table
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)
        return self._mask_input_cols(self._full_test_table)


class ConstructorPointsTask(RelBenchTask):
    r"""Predict the finishing position of each driver in a race."""
    name = "rel-f1-points-constructor"
    task_type = (
        TaskType.REGRESSION
    )  # TaskType.BINARY_CLASSIFICATION # TaskType.REGRESSION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "points"
    timedelta = pd.Timedelta(days=30)
    metrics = [mae, rmse]  # [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for results_position_next_race."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        constructors = db.table_dict["constructors"].df
        constructor_results = db.table_dict["constructor_results"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    con.constructorId as constructorId,
                    sum(re.points) points
                FROM
                    timestamp_df t
                LEFT JOIN
                    constructor_results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    constructors con
                ON
                    re.constructorId = con.constructorId
                WHERE
                    con.constructorId IN (
                        SELECT DISTINCT constructorId
                        FROM constructor_results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, con.constructorId

            ;
            """
        ).df()

        # make into binary classification task
        # df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 0. else 0)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if "val" not in self._cached_table_dict:
            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    self.dataset.val_timestamp,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["val"] = table
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if "full_test" not in self._cached_table_dict:
            full_table = self.make_table(
                self.dataset._full_db,
                pd.date_range(
                    self.dataset.test_timestamp,
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["full_test"] = full_table
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)
        return self._mask_input_cols(self._full_test_table)


class DidNotFinishTask(RelBenchTask):
    r"""Predict the if each driver will DNF (not finish) a race in the next time period."""
    name = "rel-f1-dnf"
    task_type = TaskType.BINARY_CLASSIFICATION  # TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "did_not_finish"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]  # [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for results_position_next_race."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    CASE
                        WHEN MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) = 1 THEN 0
                        ELSE 1
                    END AS did_not_finish
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        # make into binary classification task
        df[self.target_col] = df[self.target_col].apply(lambda x: 1 if x > 0.0 else 0)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""
        if "val" not in self._cached_table_dict:
            table = self.make_table(
                self.dataset.db,
                pd.date_range(
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    self.dataset.val_timestamp,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["val"] = table
        else:
            table = self._cached_table_dict["val"]
        return self.filter_dangling_entities(table)

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""
        if "full_test" not in self._cached_table_dict:
            full_table = self.make_table(
                self.dataset._full_db,
                pd.date_range(
                    self.dataset.test_timestamp,
                    self.dataset.test_timestamp
                    - (self.dataset.test_timestamp - self.dataset.val_timestamp) / 2
                    - self.timedelta,
                    freq=-self.timedelta,
                ),
            )
            self._cached_table_dict["full_test"] = full_table
        else:
            full_table = self._cached_table_dict["full_test"]
        self._full_test_table = self.filter_dangling_entities(full_table)
        return self._mask_input_cols(self._full_test_table)

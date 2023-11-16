import os

from typing import Dict, Tuple
import pandas as pd

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset
from rtb.utils import to_unix_time


class investigator_three_years(Task):
    r"""Predict the sum of grant amount in the next 3 years for an investigator."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "email_id"],
            target_col="award_sum",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta(days=3 * 365.25)],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for grant_three_years."""

        awards = db.tables["awards"].df
        investigator_awards = db.tables["investigator_awards"].df
        investigator_awards = investigator_awards[
            investigator_awards.email_id.notnull()
        ]  # 89367/635810 have missing email_ids
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                email_id,
                SUM(award_amount) AS award_sum
            FROM
                time_window_df,
                (
                    SELECT
                        email_id,
                        award_effective_date,
                        award_amount
                    FROM
                        awards,
                        investigator_awards
                    WHERE
                        awards.award_id = investigator_awards.award_id
                ) AS tmp
            WHERE
                tmp.award_effective_date > time_window_df.window_min_time AND
                tmp.award_effective_date <= time_window_df.window_max_time
            GROUP BY email_id, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"email_id": "investigator"},
            pkey_col=None,
            time_col="window_min_time",
        )


class institution_one_year(Task):
    r"""Predict the sum of grant amount in the next year for an institution."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "name"],
            target_col="award_sum",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta(days=365.25)],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for institution_one_year."""

        awards = db.tables["awards"].df
        institution_awards = db.tables["institution_awards"].df
        # institution_awards = institution_awards[institution_awards.name.notnull()] # zero null
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                name,
                SUM(award_amount) AS award_sum
            FROM
                time_window_df,
                (
                    SELECT
                        name,
                        award_effective_date,
                        award_amount
                    FROM
                        awards,
                        institution_awards
                    WHERE
                        awards.award_id = institution_awards.award_id
                ) AS tmp
            WHERE
                tmp.award_effective_date > time_window_df.window_min_time AND
                tmp.award_effective_date <= time_window_df.window_max_time
            GROUP BY name, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"name": "institution"},
            pkey_col=None,
            time_col="window_min_time",
        )


class program_three_years(Task):
    r"""Predict the sum of grant amount in the next three year for a program (research direction)."""

    def __init__(self):
        super().__init__(
            input_cols=["window_min_time", "window_max_time", "code"],
            target_col="award_sum",
            task_type=TaskType.REGRESSION,
            window_sizes=[pd.Timedelta(days=3 * 365.25)],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for program_three_years."""

        awards = db.tables["awards"].df
        program_element_awards = db.tables["program_element_awards"].df
        # program_element_awards = program_element_awards[program_element_awards.name.notnull()] # zero null
        import duckdb

        df = duckdb.sql(
            r"""
            SELECT
                window_min_time,
                window_max_time,
                code,
                SUM(award_amount) AS award_sum
            FROM
                time_window_df,
                (
                    SELECT
                        code,
                        award_effective_date,
                        award_amount
                    FROM
                        awards,
                        program_element_awards
                    WHERE
                        awards.award_id = program_element_awards.award_id
                ) AS tmp
            WHERE
                tmp.award_effective_date > time_window_df.window_min_time AND
                tmp.award_effective_date <= time_window_df.window_max_time
            GROUP BY code, window_min_time, window_max_time
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"code": "program_element"},
            pkey_col=None,
            time_col="window_min_time",
        )


class GrantDataset(Dataset):
    name = "rtb-grant"

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""
        tasks = {
            "investigator_three_years": investigator_three_years(),
            "institution_one_year": institution_one_year(),
            "program_three_years": program_three_years(),
        }

        self.tasks_window_size = {i: j.window_sizes[0] for i, j in tasks.items()}
        return tasks

    def process(self) -> Database:
        r"""Process the raw files into a database."""
        path = f"{self.root}/{self.name}/raw/"
        foa_info_awards = pd.read_csv(os.path.join(path, "foa_info_awards.csv"))
        foa_info = pd.read_csv(os.path.join(path, "foa_info.csv"))
        institution_awards = pd.read_csv(os.path.join(path, "institution_awards.csv"))
        investigator_awards = pd.read_csv(os.path.join(path, "investigator_awards.csv"))
        institution = pd.read_csv(os.path.join(path, "institution.csv"))
        awards = pd.read_csv(os.path.join(path, "awards.csv"))
        organization = pd.read_csv(os.path.join(path, "organization.csv"))
        investigator = pd.read_csv(os.path.join(path, "investigator.csv"))
        program_element = pd.read_csv(os.path.join(path, "program_element.csv"))
        program_reference = pd.read_csv(os.path.join(path, "program_reference.csv"))
        program_element_awards = pd.read_csv(
            os.path.join(path, "program_element_awards.csv")
        )
        program_reference_awards = pd.read_csv(
            os.path.join(path, "program_reference_awards.csv")
        )

        ## turn date to unix time
        investigator_awards["start_date"] = to_unix_time(
            investigator_awards["start_date"]
        )
        awards["award_effective_date"] = to_unix_time(awards["award_effective_date"])

        ## set to 1976 forward
        # print('length of investigator_awards table, ', len(investigator_awards)) 635810
        # print('length of awards table, ', len(awards)) 430339

        investigator_awards = investigator_awards[
            investigator_awards.start_date > pd.Timestamp("1976-01-01")
        ].reset_index(drop=True)
        awards = awards[
            awards.award_effective_date > pd.Timestamp("1976-01-01")
        ].reset_index(drop=True)
        # print('after 1976, length of investigator_awards table, ', len(investigator_awards)) 620021
        # print('after 1976, length of awards table, ', len(awards)) 415838

        ## add pkey columns

        institution["institution_name"] = institution["name"]
        investigator["email"] = investigator["email_id"]

        ## for each table, drop duplicated pkeys
        investigator = investigator[investigator.email_id.notnull()].reset_index(
            drop=True
        )  ## 3 have NaN email_ids, duplicating rows

        tables = {}

        tables["foa_info_awards"] = Table(
            df=pd.DataFrame(foa_info_awards),
            fkey_col_to_pkey_table={
                "award_id": "awards",
                "code": "foa_info",
            },
            pkey_col=None,
            time_col=None,
        )

        tables["foa_info"] = Table(
            df=pd.DataFrame(foa_info),
            fkey_col_to_pkey_table={},
            pkey_col="code",
            time_col=None,
        )

        tables["institution_awards"] = Table(
            df=pd.DataFrame(institution_awards),
            fkey_col_to_pkey_table={
                "name": "institution",
                "award_id": "awards",
            },
            pkey_col=None,
            time_col=None,
        )

        tables["institution"] = Table(
            df=pd.DataFrame(institution),
            fkey_col_to_pkey_table={},
            pkey_col="name",
            time_col=None,
        )

        tables["investigator_awards"] = Table(
            df=pd.DataFrame(investigator_awards),
            fkey_col_to_pkey_table={
                "email_id": "investigator",
                "award_id": "awards",
            },
            pkey_col=None,
            time_col="start_date",
        )

        tables["awards"] = Table(
            df=pd.DataFrame(awards),
            fkey_col_to_pkey_table={"organisation_code": "organization"},
            pkey_col="award_id",
            time_col="award_effective_date",
        )

        tables["organization"] = Table(
            df=pd.DataFrame(organization),
            fkey_col_to_pkey_table={},
            pkey_col="code",
            time_col=None,
        )

        tables["investigator"] = Table(
            df=pd.DataFrame(investigator),
            fkey_col_to_pkey_table={},
            pkey_col="email_id",
            time_col=None,
        )

        tables["program_element"] = Table(
            df=pd.DataFrame(program_element),
            fkey_col_to_pkey_table={},
            pkey_col="code",
            time_col=None,
        )

        tables["program_reference"] = Table(
            df=pd.DataFrame(program_reference),
            fkey_col_to_pkey_table={},
            pkey_col="code",
            time_col=None,
        )

        tables["program_element_awards"] = Table(
            df=pd.DataFrame(program_element_awards),
            fkey_col_to_pkey_table={
                "code": "program_element",
                "award_id": "awards",
            },
            pkey_col=None,
            time_col=None,
        )

        tables["program_reference_awards"] = Table(
            df=pd.DataFrame(program_reference_awards),
            fkey_col_to_pkey_table={
                "code": "program_reference",
                "award_id": "awards",
            },
            pkey_col=None,
            time_col=None,
        )

        return Database(tables)

    def get_cutoff_times(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        # train_max_time = 1293782400 ## year 2010-12-31 (3+3 years before max time)
        # val_max_time = 1388476800 ## year 2013-12-31 (3 years before max time)
        train_max_time = pd.Timestamp("2010-12-31")
        val_max_time = pd.Timestamp("2013-12-31")
        return train_max_time, val_max_time

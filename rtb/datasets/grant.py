import json
import os
import re

from typing import Dict, Union, Tuple
import duckdb
import pandas as pd
from tqdm.auto import tqdm

from rtb.data.table import Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset
from rtb.utils import to_unix_time


class grant_three_years(Task):
    r"""Predict the sum of grant amount in the next 3 years for an investigator."""

    def __init__(self):
        super().__init__(
            target_col="grant_three_years",
            task_type=TaskType.REGRESSION,
            test_time_window_sizes=[3 * 365 * 24 * 60 * 60],
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for grant_three_years."""

        awards = db["awards"].df
        investigator_awards = db["investigator_awards"].df
        investigator_awards = investigator_awards[investigator_awards.email_id.notnull()] # 89367/635810 have missing email_ids
        import duckdb
        df = duckdb.sql(
            r"""
            SELECT
                time_offset,
                time_cutoff,
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
                tmp.award_effective_date > time_window_df.time_offset AND
                tmp.award_effective_date <= time_window_df.time_cutoff
            GROUP BY email_id, time_offset, time_cutoff
            """
        ).df()


        return Table(
            df=df,
            fkeys={"email_id": "investigator"},
            pkey=None,
            time_col="time_offset",
        )

class GrantDataset(Dataset):

    name = "rtb-grant"

    def get_tasks(self) -> Dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"grant_three_years": grant_three_years()}


    def download(self, path: Union[str, os.PathLike]) -> None:

        """
        Download a file from an S3 bucket.
        Parameters:
        - path (str): Local path where the file should be saved.
        Returns:
        None
        
        file_key = f"{self.root}/{self.name}"
        bucket_name = 'XXX' ## TBD
        region_name='us-west-2' ## TBD
        # Create an S3 client
        s3 = boto3.client('s3', region_name=region_name)
        # Download the file
        s3.download_file(bucket_name, file_key, path)   
        """
        pass

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
        program_element_awards = pd.read_csv(os.path.join(path, "program_element_awards.csv"))
        program_reference_awards = pd.read_csv(os.path.join(path, "program_reference_awards.csv"))

        ## turn date to unix time
        investigator_awards["start_date"] = to_unix_time(investigator_awards["start_date"])
        awards["award_effective_date"] = to_unix_time(awards["award_effective_date"])

        ## set to 1976 forward
        #print('length of investigator_awards table, ', len(investigator_awards)) 635810
        #print('length of awards table, ', len(awards)) 430339
       
        investigator_awards = investigator_awards[investigator_awards.start_date > pd.Timestamp('1976-01-01')].reset_index(drop = True)
        awards = awards[awards.award_effective_date > pd.Timestamp('1976-01-01')].reset_index(drop = True)
        #print('after 1976, length of investigator_awards table, ', len(investigator_awards)) 620021
        #print('after 1976, length of awards table, ', len(awards)) 415838

        
        tables = {}

        tables["foa_info_awards"] = Table(
            df=pd.DataFrame(foa_info_awards),
            fkeys={
                "award_id": "awards",
                "code": "foa_info",
            },
            pkey=None,
            time_col=None,
        )


        tables["foa_info"] = Table(
            df=pd.DataFrame(foa_info),
            fkeys={},
            pkey="code",
            time_col=None,
        )


        tables["institution_awards"] = Table(
            df=pd.DataFrame(institution_awards),
            fkeys={
                "name": "institution",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )

        tables["institution"] = Table(
            df=pd.DataFrame(institution),
            fkeys={},
            pkey="name",
            time_col=None,
        )


        tables["investigator_awards"] = Table(
            df=pd.DataFrame(investigator_awards),
            fkeys={
                "email_id": "investigator",
                "award_id": "awards",
            },
            pkey=None,
            time_col="start_date",
        )


        tables["awards"] = Table(
            df=pd.DataFrame(awards),
            fkeys={
                "organisation_code": "organization"
            },
            pkey="award_id",
            time_col="award_effective_date",
        )

        tables["organization"] = Table(
            df=pd.DataFrame(organization),
            fkeys={},
            pkey="code",
            time_col=None,
        )

        tables["investigator"] = Table(
            df=pd.DataFrame(investigator),
            fkeys={},
            pkey="email_id",
            time_col=None,
        )

        tables["program_element"] = Table(
            df=pd.DataFrame(program_element),
            fkeys={},
            pkey="code",
            time_col=None,
        )

        tables["program_reference"] = Table(
            df=pd.DataFrame(program_reference),
            fkeys={},
            pkey="code",
            time_col=None,
        )

        tables["program_element_awards"] = Table(
            df=pd.DataFrame(program_element_awards),
            fkeys={
                "code": "program_element",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )

        tables["program_reference_awards"] = Table(
            df=pd.DataFrame(program_reference_awards),
            fkeys={
                "code": "program_reference",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )

        return Database(tables)

    def get_cutoff_times(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        #train_cutoff_time = 1293782400 ## year 2010-12-31 (3+3 years before max time)
        #val_cutoff_time = 1388476800 ## year 2013-12-31 (3 years before max time)
        train_cutoff_time = pd.Timestamp('2010-12-31')
        val_cutoff_time = pd.Timestamp('2013-12-31')
        return train_cutoff_time, val_cutoff_time
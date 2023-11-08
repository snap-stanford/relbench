import json
import os
import re

import duckdb
import pandas as pd
from tqdm.auto import tqdm

from rtb.data.table import SemanticType, Table
from rtb.data.database import Database
from rtb.data.task import TaskType, Task
from rtb.data.dataset import Dataset


class grant_five_years(Task):
    r"""Predict the sum of grant amount in the next 5 years for an investigator."""

    def __init__(self):
        super().__init__(
            target_col="grant_five_years",
            task_type=TaskType.REGRESSION,
            metrics=["mse", "smape"],
        )

    def make_table(self, db: Database, time_window_df: pd.DataFrame) -> Table:
        r"""Create Task object for LTV."""

        pass


class GrantDataset(Dataset):

    name = "rtb-grant"

    def get_tasks(self) -> dict[str, Task]:
        r"""Returns a list of tasks defined on the dataset."""

        return {"grant_five_years": grant_five_years()}

    # TODO: implement get_cutoff_times()

    def download(self, path: str | os.PathLike) -> None:
        
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
        
        tables = {}

        tables["foa_info_awards"] = Table(
            df=pd.DataFrame(foa_info_awards),
            feat_cols={},
            fkeys={
                "award_id": "awards",
                "code": "foa_info",
            },
            pkey=None,
            time_col=None,
        )
        
        
        tables["foa_info"] = Table(
            df=pd.DataFrame(foa_info),
            feat_cols={
                "name": SemanticType.TEXT,
            },
            fkeys={},
            pkey="code",
            time_col=None,
        )
        
        
        tables["institution_awards"] = Table(
            df=pd.DataFrame(institution_awards),
            feat_cols={
                "zipcode": SemanticType.TEXT,
            },
            fkeys={
                "name": "institution",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )
        
        tables["institution"] = Table(
            df=pd.DataFrame(institution),
            feat_cols={
                "city_name": SemanticType.TEXT,
                "zipcode": SemanticType.TEXT,
                "contact": SemanticType.NUMERICAL,
                "address": SemanticType.TEXT,
                "country_name": SemanticType.TEXT,
                "state_name": SemanticType.TEXT,
                "state_code": SemanticType.TEXT,
            },
            fkeys={},
            pkey="name",
            time_col=None,
        )
        
        
        tables["investigator_awards"] = Table(
            df=pd.DataFrame(investigator_awards),
            feat_cols={
                "start_date": SemanticType.Time,
                "end_date": SemanticType.Time, ## ?? 
                "role_code": SemanticType.TEXT
            },
            fkeys={
                "email_id": "investigator",
                "award_id": "awards",
            },
            pkey=None,
            time_col="start_date",
        )
        
        
        tables["awards"] = Table(
            df=pd.DataFrame(awards),
            feat_cols={
                "award_title": SemanticType.TEXT,
                "award_effective_date": SemanticType.Time,
                "award_expiration_date": SemanticType.Time,
                "award_amount": SemanticType.NUMERICAL,
                "award_instrument": SemanticType.TEXT,
                "program_officer": SemanticType.TEXT,
                "abstract_narration": SemanticType.TEXT,
                "min_amd_letter_date": SemanticType.Time,
                "max_amd_letter_date": SemanticType.Time,
                "arra_amount": SemanticType.NUMERICAL
            },
            fkeys={
                "organisation_code": "organization" ## does the column name in foreign key table have the same name as the primary key column?
            },
            pkey="award_id",
            time_col="award_effective_date",
        )
        
        tables["organization"] = Table(
            df=pd.DataFrame(organization),
            feat_cols={
                "division": SemanticType.TEXT,
                "directorate": SemanticType.TEXT
            },
            fkeys={},
            pkey="code",
            time_col=None,
        )
        
        tables["investigator"] = Table(
            df=pd.DataFrame(investigator),
            feat_cols={
                "first_name": SemanticType.TEXT,
                "last_name": SemanticType.TEXT
            },
            fkeys={},
            pkey="email_id",
            time_col=None,
        )
        
        tables["program_element"] = Table(
            df=pd.DataFrame(program_element),
            feat_cols={
                "text": SemanticType.TEXT,
            },
            fkeys={},
            pkey="code",
            time_col=None,
        )
        
        tables["program_reference"] = Table(
            df=pd.DataFrame(program_reference),
            feat_cols={
                "text": SemanticType.TEXT,
            },
            fkeys={},
            pkey="code",
            time_col=None,
        )
        
        tables["program_element_awards"] = Table(
            df=pd.DataFrame(program_element_awards),
            feat_cols={},
            fkeys={
                "code": "program_element",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )
        
        tables["program_reference_awards"] = Table(
            df=pd.DataFrame(program_reference_awards),
            feat_cols={},
            fkeys={
                "code": "program_reference",
                "award_id": "awards",
            },
            pkey=None,
            time_col=None,
        )

        return Database(tables)
    
    def get_cutoff_times(self) -> tuple[int, int]:
        r"""Returns the train and val cutoff times. To be implemented by
        subclass, but can implement a sensible default strategy here."""

        train_cutoff_time = 2010
        val_cutoff_time = 2012
        return train_cutoff_time, val_cutoff_time
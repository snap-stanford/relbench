import os

import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.f1 import PointsTask
from relbench.utils import unzip_processor

class F1Dataset(RelBenchDataset):

    name = "rel-f1"
    val_timestamp = pd.Timestamp("2000-01-01")
    test_timestamp = pd.Timestamp("2010-01-01")
    task_cls_list = [PointsTask]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        path = "relbench/datasets/f1-temp" # temporary path for development

        path = os.path.join(path, "raw")

        print("Current working directory:", os.getcwd())

        circuits = pd.read_csv(os.path.join(path, "circuits.csv"))
        drivers = pd.read_csv(os.path.join(path, "drivers.csv"))
        results = pd.read_csv(os.path.join(path, "results.csv"))
        races = pd.read_csv(os.path.join(path, "races.csv"))
        standings = pd.read_csv(os.path.join(path, "driver_standings.csv"))


        ## remove columns that are irrelevant, leak time, or have too many missing values
        races.drop(
            columns=["url",
                     "fp1_date",
                     "fp1_time",
                     "fp2_date",
                     "fp2_time",
                     "fp3_date",
                     "fp3_time",
                     "quali_date",
                     "quali_time",
                     "sprint_date",
                     "sprint_time"
            ],
            inplace=True,
        )

        circuits.drop(
            columns=["url"],
            inplace=True,
        )


        drivers.drop(
            columns=["number", "url"],
            inplace=True,
        )

        results.drop(
            columns=["constructorId", "statusId"],
            inplace=True,
        )


        standings.drop(
            columns=["positionText"],
            inplace=True,
        )


        ## replase missing data and combine date and time columns
        races['time'] = races['time'].replace(r'^\\N$', '00:00:00', regex=True)
        races["date"] = races['date'] + ' ' + races['time']
        ## change time column to unix time
        races["date"] = pd.to_datetime(races["date"])

        # add time column to results table
        results = results.merge(races[['raceId', 'date']], on='raceId', how='left')

        # add time column to standings table
        standings = standings.merge(races[['raceId', 'date']], on='raceId', how='left')


        tables = {}

        tables["races"] = Table(
            df=pd.DataFrame(races),
            fkey_col_to_pkey_table={
                "circuitId": "circuits",  
            },
            pkey_col="raceId",
            time_col="date",
        )

        tables["circuits"] = Table(
            df=pd.DataFrame(circuits),
            fkey_col_to_pkey_table={},
            pkey_col="circuitId",
            time_col=None,
        )

        tables["drivers"] = Table(
            df=pd.DataFrame(drivers),
            fkey_col_to_pkey_table={},
            pkey_col="driverId",
            time_col=None,
        )

        tables["results"] = Table(
            df=pd.DataFrame(results),
            fkey_col_to_pkey_table={
                "raceId": "races", 
                "driverId": "drivers"},
            pkey_col="resultId",
            time_col="date",
        )

        tables["standings"] = Table(
            df=pd.DataFrame(standings),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers"},
            pkey_col="driverStandingsId",
            time_col="date" 
        )
 
        return Database(tables)

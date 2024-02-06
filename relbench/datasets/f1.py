import os

import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.f1 import PointsTask, ConstructorPointsTask, DidNotFinishTask, PodiumTask
from relbench.utils import unzip_processor

class F1Dataset(RelBenchDataset):

    name = "rel-f1"
    val_timestamp = pd.Timestamp("2000-01-01")
    test_timestamp = pd.Timestamp("2015-01-01")
    task_cls_list = [PointsTask, ConstructorPointsTask, DidNotFinishTask, PodiumTask]

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
        constructors = pd.read_csv(os.path.join(path, "constructors.csv"))
        constructor_results = pd.read_csv(os.path.join(path, "constructor_results.csv"))
        constructor_standings = pd.read_csv(os.path.join(path, "constructor_standings.csv"))
        lap_times = pd.read_csv(os.path.join(path, "lap_times.csv"))
        qualifying = pd.read_csv(os.path.join(path, "qualifying.csv"))  
        sprint_results = pd.read_csv(os.path.join(path, "sprint_results.csv"))
        pit_stops = pd.read_csv(os.path.join(path, "pit_stops.csv"))

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
                     "sprint_time",
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
            columns=["positionText"],
            inplace=True,
        )

        standings.drop(
            columns=["positionText"],
            inplace=True,
        )

        constructors.drop(
            columns=["url"],
            inplace=True,
        )

        constructor_standings.drop(
            columns=["positionText"],
            inplace=True,
        )

        lap_times.drop(
            columns=["time"],
            inplace=True,
        )

        qualifying.drop(
            columns=["q1", "q2", "q3"],
            inplace=True,
        )

        sprint_results.drop(
            columns=["positionText"],
            inplace=True,
        )

        pit_stops.drop(
            columns=["time"],
            inplace=True,
        )


        ## replase missing data and combine date and time columns
        races['time'] = races['time'].replace(r'^\\N$', '00:00:00', regex=True)
        races["date"] = races['date'] + ' ' + races['time']
        ## change time column to unix time
        races["date"] = pd.to_datetime(races["date"])

        # add time column to other tables
        results = results.merge(races[['raceId', 'date']], on='raceId', how='left')
        standings = standings.merge(races[['raceId', 'date']], on='raceId', how='left')
        constructor_results = constructor_results.merge(races[['raceId', 'date']], on='raceId', how='left')
        constructor_standings = constructor_standings.merge(races[['raceId', 'date']], on='raceId', how='left')
        lap_times = lap_times.merge(races[['raceId', 'date']], on='raceId', how='left')

        qualifying = qualifying.merge(races[['raceId', 'date']], on='raceId', how='left')
        # subtract a day from the date to account for the fact 
        # 
        # that the qualifying time is the day before the main race
        qualifying['date'] = qualifying['date'] - pd.Timedelta(days=1)

        sprint_results = sprint_results.merge(races[['raceId', 'date']], on='raceId', how='left')
        # rename resultId to sprintResultId to distinguish from results
        sprint_results.rename(columns={'resultId': 'sprintResultId'}, inplace=True)

        pit_stops = pit_stops.merge(races[['raceId', 'date']], on='raceId', how='left')
        # add id column to pit_stops
        pit_stops['pitStopId'] = pit_stops.index

        # add missing pkey colum to lap_times
        lap_times['lapId'] = lap_times.index

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

        tables["constructors"] = Table(
            df=pd.DataFrame(constructors),
            fkey_col_to_pkey_table={},
            pkey_col="constructorId",
            time_col=None,
        )

        tables["constructor_results"] = Table(
            df=pd.DataFrame(constructor_results),
            fkey_col_to_pkey_table={
                'raceId': 'races',
                'constructorId': 'constructors'},
            pkey_col="constructorResultsId",
            time_col="date"
        )

        tables["constructor_standings"] = Table(
            df=pd.DataFrame(constructor_standings),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "constructorId": "constructors"},
            pkey_col="constructorStandingsId",
            time_col="date"
        )

        tables["lap_times"] = Table(
            df=pd.DataFrame(lap_times),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers"},
            pkey_col="lapId",
            time_col="date"
        )

        tables["qualifying"] = Table(
            df=pd.DataFrame(qualifying),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors"},
            pkey_col="qualifyId",
            time_col="date"
        )


        """
        tables["sprint_results"] = Table(
            df=pd.DataFrame(sprint_results),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors"},
            pkey_col="sprintResultId",
            time_col="date"
        )

        tables["pit_stops"] = Table(
            df=pd.DataFrame(pit_stops),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers"},
            pkey_col="pitStopId",
            time_col="date"
        )
        """
        return Database(tables)

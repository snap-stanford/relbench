import copy
import datetime
import decimal
import os
import time
from pathlib import Path

import pandas as pd
import sqlparse
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from relbench.base import Database, Dataset, Table


def find_time_col(columns):
    for col in [
        "admittime",
        "startdate",
        "chartdate",
        "transfertime",
        "charttime",
        "intime",
        "starttime",
    ]:
        if col in columns:
            return col
    return None


def format_ids(series):
    return f"({', '.join(map(str, series.dropna().astype(int).unique()))})"


def is_not_numeric_string(x):
    try:
        float(x)
        return False
    except (ValueError, TypeError):
        return True


def get_tables_with_schema_query(schema):
    tables_query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema}';
    """
    return tables_query


def filter_chartevents(table):
    print("Filtering chartevents to only include numeric values")
    chart_df = table.df

    numeric_mask = chart_df["valuenum"].notnull() & (
        chart_df["value"].astype(str) == chart_df["valuenum"].astype(str)
    )
    chartevents_numeric = chart_df[numeric_mask].copy()
    chartevents_numeric = chartevents_numeric.drop(columns=["value"])
    chartevents_numeric["valuenum"] = pd.to_numeric(
        chartevents_numeric["valuenum"], errors="coerce"
    )
    return Table(
        df=chartevents_numeric,
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


class MimicDataset(Dataset):
    """A dataset class for loading and processing MIMIC-IV data into a format suitable
    for use with RelBench. This class extracts a subset of tables specified in
    `tables_limit`, filters the data based on ICU stay and patient criteria, and allows
    optional column dropouts to reduce dimensionality or remove irrelevant fields.

    The dataset is anchored around ICU stays (`icustays`) and patients (`patients`) and constructs all other
    tables with respect to these entities.

    Parameters:
        patients_limit (int): Maximum number of patients to include in the dataset (0 means no limit).
        saving_data (bool): Whether to persist processed tables to disk as .H5 files.
        out_path (str): Output directory to save HDF5 files.
        cache_dir (str): Directory used for caching dataset.
        tables_limit (list): List of table names to include in the dataset.
        drop_columns_per_table (dict): Dictionary specifying columns to drop per table.
        min_age (int): Minimum patient age in years to include.
        min_dur (int): Minimum ICU stay duration in hours.
        max_dur (int): Maximum ICU stay duration in hours.
        db_params (dict): Dictionary with database connection parameters.

    Example:
        drop_columns_per_table = {
            "admissions": [...],
            "chartevents": [...],
            ...
        }
        tables_limit = ["patients", "admissions", "icustays", "chartevents", "procedureevents", "d_items"]

        dataset = MimicDataset(
            patients_limit=20 000,
            out_path='/data',
            cache_dir='/cache',
            tables_limit=tables_limit,
            db_params=db_params,
            drop_columns_per_table=drop_columns_per_table
        )
        db = dataset.make_db()
    """

    schema_name = "public,public,mimiciv_derived, mimiciv_hosp, mimiciv_icu, mimiciv_ed"

    def __init__(
        self,
        patients_limit: int = -1,
        saving_data: bool = True,
        out_path: str = "data",
        cache_dir: str = None,
        tables_limit: list = None,
        drop_columns_per_table=None,
        min_age: int = 15,
        min_dur: int = 36,
        max_dur: int = 240,
        db_params: dict = None,
    ):
        if drop_columns_per_table is None:
            print(f"drop_columns_per_table not provided, dropping default.")
            drop_columns_per_table = {
                "admissions": [
                    "admittime",
                    "dischtime",
                    "deathtime",
                    "edouttime",
                    "edregtime",
                    "hospital_expire_flag",
                    "discharge_location",
                    "race",
                ],
                "chartevents": ["storetime"],
                "procedureevents": [
                    "storetime",
                    "endtime",
                    "location",
                    "location_category",
                    "location",
                    "locationcategory",
                    "linkorderid",
                    "originalamount",
                    "originalrate",
                ],
                "d_items": ["abbreviation"],
            }
        if db_params is None:
            load_dotenv()
            print(f"db_params not provided — loading from .env file.")
            db_params = {
                "dbname": os.getenv("DB_NAME"),
                "host": os.getenv("DB_HOST"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "port": os.getenv("DB_PORT"),
            }
        if tables_limit is None:
            print(f"tables_limit not provided setting to default.")
            tables_limit = [
                "patients",
                "admissions",
                "icustays",
                "chartevents",
                "procedureevents",
                "d_items",
            ]

        if patients_limit == -1:
            patients_limit = 1000
            print(f"patients_limit not provided setting to default {patients_limit}.")
        print("Cachce dir", end=" ")
        super().__init__(cache_dir=cache_dir)

        self.query_args = {
            "limit": patients_limit,
            "min_age": str(min_age),
            "min_dur": str(min_dur),
            "max_dur": str(max_dur),
            "min_day": str(float(min_dur) / 24),
        }

        self.saving_data = saving_data
        self.patients_limit = patients_limit
        self.drop_columns_per_table = drop_columns_per_table

        self.tables_limit = tables_limit

        # Create the output directory if it doesn't exist
        current_dir = os.getcwd()
        self.out_path = Path(current_dir) / out_path / f"limit_{patients_limit}"
        if saving_data:
            self.out_path.mkdir(parents=True, exist_ok=True)
            print("Data will be saved to", self.out_path)

        # Initialize the querier
        self.connected = False
        self.engine = None
        self.session = None
        self.connect(db_params=db_params)

        # Set the test and validation timestamps
        # this will be changed in the based on patients limit
        self.test_timestamp = pd.Timestamp("2180-10-26 14:53:54")
        self.val_timestamp = pd.Timestamp("2168-09-06 07:13:00")

    def make_db(self) -> Database:
        start_time = time.time()
        tables_df = {}

        tables_df["patients"] = Table(
            df=self.get_patients(),
            fkey_col_to_pkey_table=self.get_foreign_keys("patients", "mimiciv_hosp"),
            pkey_col=self.get_primary_key("patients", "mimiciv_hosp"),
            time_col=None,
        )
        template_vars = dict(
            subject_ids=format_ids(tables_df["patients"].df["subject_id"]),
            stay_ids=format_ids(tables_df["patients"].df["stay_id"]),
            hadm_ids=format_ids(tables_df["patients"].df["hadm_id"]),
        )

        tables_df["icustays"] = Table(
            df=self.get_icustays(template_vars),
            fkey_col_to_pkey_table=self.get_foreign_keys("icustays", "mimiciv_icu"),
            pkey_col=self.get_primary_key("icustays", "mimiciv_icu"),
            time_col="intime",
        )

        for schema, tables in self.get_filtered_tables().items():
            for (table_name,) in tables:
                if table_name in {"patients", "icustays"}:
                    continue
                h5_path = self.out_path / f"{table_name}_{self.patients_limit}.H5"
                if not h5_path.exists():
                    print(f"Creating {table_name}")

                    # Create the query string
                    query_string = self.build_query(table_name, schema)

                    # Update template_vars with the current table name and schema
                    template_vars["schema"] = schema
                    template_vars["table_name"] = table_name

                    print("Querying table:", table_name, end=" ")
                    # Execute the query
                    df = self.query(
                        query_string=query_string, template_vars=template_vars
                    )
                    print(f"{len(df)} rows")

                    # Drop columns if specified
                    if (
                        self.drop_columns_per_table
                        and table_name in self.drop_columns_per_table
                    ):
                        df = self.drop_columns(table_name, df)

                    # Save the DataFrame to HDF5 if saving_data is True
                    if self.saving_data:
                        df.to_hdf(h5_path, key="table", index=False)
                        print(f"Table {table_name} saved to {h5_path}")
                else:
                    print(
                        f"️File {table_name}_{self.patients_limit}.H5 already exists. Skipping..."
                    )
                    df = pd.read_hdf(h5_path, key="table")

                # Create the Table object for the current table
                tables_df[table_name] = Table(
                    df=df,
                    fkey_col_to_pkey_table=self.get_foreign_keys(table_name, schema),
                    pkey_col=self.get_primary_key(table_name, schema),
                    time_col=find_time_col(df.columns),
                )

        # Filter chartevents to only include numeric values
        tables_df["chartevents"] = filter_chartevents(tables_df["chartevents"])

        # Change test and val timestamps based on patients limit
        print("Setting test and val timestamps based on patients limit")
        timestamps = (
            pd.to_datetime(tables_df["icustays"].df["intime"], errors="coerce")
            .dropna()
            .sort_values()
        )
        n = len(timestamps)
        # Calculate the indices for 70% and 85%
        # of the total number of timestamps
        # 70% for train and 15% for validation and 15% for test
        val_idx = int(n * 0.7)
        test_idx = int(n * 0.85)
        self.val_timestamp = timestamps.iloc[val_idx]
        self.test_timestamp = timestamps.iloc[test_idx]
        train_count = len(timestamps[timestamps < self.val_timestamp])
        val_count = len(
            timestamps[
                (timestamps >= self.val_timestamp) & (timestamps < self.test_timestamp)
            ]
        )
        test_count = len(timestamps[timestamps >= self.test_timestamp])
        print("test_timestamp:", self.test_timestamp, end=" ")
        print("val_timestamp:", self.val_timestamp)
        print(
            "Record split - Train:", train_count, "Val:", val_count, "Test:", test_count
        )

        end_time = time.time()
        print(f"Done loading tables in {end_time - start_time:.2f} seconds")
        return Database(tables_df)

    def get_patients(self):
        patients_filename = "patients_" + str(self.patients_limit) + ".H5"
        # Query the patients
        patients_query = """
         SELECT DISTINCT ON (i.subject_id)
             i.subject_id,                  -- Unique patient identifier
            i.hadm_id,                     -- Unique hospital admission ID
            i.stay_id,                     -- Unique ICU stay ID
            i.gender,                      -- Patient gender
            ROUND(i.admission_age) AS age, -- Age of the patient at hospital admission
            i.race                         -- Patient race/ethnicity
         FROM mimiciv_derived.icustay_detail i
         WHERE i.hadm_id IS NOT NULL
           AND i.stay_id IS NOT NULL
           AND i.hospstay_seq = 1
           AND i.icustay_seq = 1
           AND i.los_icu IS NOT NULL
           AND i.admission_age >= {min_age}
           AND i.los_icu >= {min_day}
           AND (i.outtime >= (i.intime + INTERVAL '{min_dur} hours'))
           AND (i.outtime <= (i.intime + INTERVAL '{max_dur} hours'))
           AND EXISTS (
             SELECT 1
             FROM mimiciv_icu.icustays icu
             WHERE icu.stay_id = i.stay_id
             )
         ORDER BY subject_id
             {limit}
         """
        if self.query_args["limit"] > 0:
            pop_size_string = "LIMIT " + str(self.query_args["limit"])
        else:
            pop_size_string = ""
        template_vars = copy.deepcopy(self.query_args)
        template_vars["limit"] = pop_size_string
        H5_fpath = os.path.join(self.out_path, patients_filename)
        if not os.path.exists(H5_fpath):
            patients = self.query(
                query_string=patients_query, template_vars=template_vars
            )
            patients["gender"] = patients["gender"].astype(str)
            patients["race"] = patients["race"].astype(str)
            patients["age"] = patients["age"].fillna(0).astype(int)
            if self.saving_data:
                patients.to_hdf(H5_fpath, key="patients", index=False)
                print(f"Patients data saved to {H5_fpath}")
        else:
            print(f"File {patients_filename} already exists. Skipping...")
            patients = pd.read_hdf(H5_fpath, key="patients")
        return patients

    def get_icustays(self, args):
        icustays_filename = "icustays_" + str(self.patients_limit) + ".H5"
        H5_fpath = os.path.join(self.out_path, icustays_filename)
        # Query the ICU_Stay
        icustays_query = """
         SELECT icu.subject_id,
                icu.hadm_id,
                icu.stay_id,
                icu.intime,
                icu.first_careunit,
                icu_detail.los_icu
         FROM mimiciv_icu.icustays AS icu
              LEFT JOIN mimiciv_derived.icustay_detail AS icu_detail
              ON icu.stay_id = icu_detail.stay_id
         WHERE icu.subject_id IN {subject_ids}
            AND icu.stay_id IN {stay_ids}
            AND icu.hadm_id IN {hadm_ids}
            AND icu_detail.los_icu IS NOT NULL
         ORDER BY icu.subject_id
         """
        if not os.path.exists(H5_fpath):

            icustays = self.query(query_string=icustays_query, template_vars=args)
            icustays["first_careunit"] = icustays["first_careunit"].astype(str)
            icustays["los_icu"] = pd.to_numeric(icustays["los_icu"], errors="coerce")
            if self.saving_data:
                icustays.to_hdf(H5_fpath, key="icustays", index=False)
                print(f"ICUStays data saved to {H5_fpath}")
        else:
            print(f"File {icustays_filename} already exists. Skipping...")
            icustays = pd.read_hdf(H5_fpath, key="icustays")

        return icustays

    def build_query(self, table_name, schema):
        columns = self.get_columns(table_name, schema)
        conditions = []
        if "subject_id" in columns:
            conditions.append("subject_id IN {subject_ids}")
        if "stay_id" in columns:
            conditions.append("stay_id IN {stay_ids}")
        if "hadm_id" in columns:
            conditions.append("hadm_id IN {hadm_ids}")

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""

        query_string = f"SELECT * FROM {{schema}}.{{table_name}}{where_clause};"
        return query_string

    def get_filtered_tables(self):
        tables_with_schema_key = {
            "mimiciv_hosp": self.query(
                query_string=get_tables_with_schema_query("mimiciv_hosp")
            ).values.tolist(),
            "mimiciv_icu": self.query(
                query_string=get_tables_with_schema_query("mimiciv_icu")
            ).values.tolist(),
        }
        return {
            schema: [t for t in tables if t[0] in self.tables_limit]
            for schema, tables in tables_with_schema_key.items()
        }

    def drop_columns(self, table_name, df):
        drop_cols = list(
            set(self.drop_columns_per_table[table_name]) & set(df.columns)
        )  # intersection
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"Dropped columns from {table_name}: {drop_cols}")
        return df

    def connect(self, db_params={}):
        if self.connected:
            return
        if db_params is None:
            print("db_params is None setting default db_params")
        else:
            print("Connecting to db with provided db_params", db_params)

        user = db_params.get("user", "postgres")
        password = db_params.get("password", "postgres")
        host = db_params.get("host", "localhost")
        dbname = db_params.get("dbname", "mimiciv")
        port = db_params.get("port", "5432")
        conn_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(conn_string)
        session = sessionmaker(bind=self.engine)
        self.session = session()
        with self.engine.connect() as conn:
            conn.execute(text(f"SET search_path TO {self.schema_name}"))
        self.connected = True

    def get_primary_key(self, table_name, schema):
        inspector = inspect(self.engine)
        referred_pk = inspector.get_pk_constraint(table_name, schema=schema)[
            "constrained_columns"
        ]
        if len(referred_pk) > 1:
            print(
                f"Warning: Table {table_name} has multiple primary keys: {referred_pk}, not supported."
            )
            return None
        return referred_pk[0] if len(referred_pk) == 1 else None

    def get_foreign_keys(self, table_name, schema):
        inspector = inspect(self.engine)
        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
        fkey_col_to_pkey_table = {}
        for fk in foreign_keys:
            fkey_col_to_pkey_table[fk["constrained_columns"][0]] = fk["referred_table"]
        return fkey_col_to_pkey_table

    def get_columns(self, table_name, schema):
        inspector = inspect(self.engine)
        columns_info = inspector.get_columns(table_name, schema=schema)
        columns = [col["name"] for col in columns_info]
        return columns

    def query(self, query_string, template_vars={}):

        def extract_table_name(query):
            parsed = sqlparse.parse(query)
            for stmt in parsed:
                from_seen = False
                schema = None
                table = None
                for token in stmt.tokens:
                    if (
                        token.ttype is sqlparse.tokens.Keyword
                        and token.value.upper() == "FROM"
                    ):
                        from_seen = True
                    elif from_seen and isinstance(token, sqlparse.sql.Identifier):
                        parts = token.get_real_name(), token.get_parent_name()
                        table, schema = parts if parts[1] else (parts[0], None)
                        return schema, table
            return None, None

        query_string = query_string.format(**template_vars)
        with self.engine.connect() as conn:
            schema, table_name = extract_table_name(query_string)
            if table_name is None:
                raise ValueError("Názov tabuľky sa nepodarilo extrahovať z query.")
            if schema is None:
                schema = self.schema_name.split(",")[
                    0
                ]  # Použije defaultnú schému, ak nebola extrahovaná
            result = conn.execute(text(query_string))
            inspector = inspect(self.engine)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            columns_info = inspector.get_columns(table_name, schema=schema)

            dtype_mapping = {col["name"]: col["type"] for col in columns_info}
            if template_vars:  # To not convert if not selecting all tables
                for col, dtype in dtype_mapping.items():
                    if col in df.columns:
                        python_type = getattr(
                            dtype, "python_type", None
                        )  # Safely get python_type
                        if python_type is datetime.datetime:
                            df[col] = pd.to_datetime(df[col])  # Conversion for datetime
                        elif python_type is datetime.date:
                            df[col] = pd.to_datetime(df[col])  # Conversion for date
                        elif python_type is decimal.Decimal:
                            df[col] = pd.to_numeric(
                                df[col]
                            )  # Conversion for decimal.Decimal
                        elif python_type is str:
                            df[col] = df[col].astype(str)  # Conversion for text
                        elif python_type is int:
                            df[col] = (
                                df[col].fillna(0).astype(int)
                            )  # Fill NaN values and convert to int
                        else:
                            df[col] = df[col].astype(python_type)

            return df


if __name__ == "__main__":
    dataset = MimicDataset(cache_dir="../../cache", patients_limit=10)
    db = dataset.make_db()
    mimic_db = dataset.get_db()

    print(mimic_db)

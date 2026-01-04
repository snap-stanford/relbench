import os
import time
from pathlib import Path

import pandas as pd

from relbench.base import Database, Dataset, Table


def verify_mimic_access() -> None:
    """Verify that the user has proper access to MIMIC-IV dataset through PhysioNet
    credentialing.

    Verification is done by attempting a small query to the dataset.
    """
    print("Verifying MIMIC-IV access...")
    try:
        from google.cloud import bigquery

        table_id = "physionet-data.mimiciv_3_1_hosp.patients"
        project = os.getenv("PROJECT_ID")
        client = bigquery.Client(project=project)
        client.get_table(table_id)
        client.query("SELECT 1").result()
        print("MIMIC-IV access verified.")
    except Exception as e:
        raise RuntimeError(
            f"\nACCESS FAILED - BigQuery credential check encountered an error: {e}"
        )


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


def format_ids(series: pd.Series) -> list:
    return list(map(int, series.dropna().unique()))


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Create a mapping for common extension dtypes
    dtype_mapping = {
        "Int64": int,
        "Int32": int,
        "Float64": float,
        "boolean": bool,
        "object": str,
        # 'string': 'object'
    }

    # Apply to all columns that have extension dtypes
    converted_dtypes = {}
    for col in df.columns:
        original_dtype = str(df[col].dtype)
        if original_dtype in dtype_mapping:
            # we need to fill NA for number conversion..
            if dtype_mapping[original_dtype] in (int, float, complex):
                df[col] = df[col].fillna(0)
            converted_dtypes[col] = dtype_mapping[original_dtype]
        elif (
            original_dtype.startswith("datetime")
            or "date" in col.lower()
            or "time" in col.lower()
        ):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                print("WARNING ")
                pass

    return df.astype(converted_dtypes)


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

    If certain parameters are not provided, default values will be used.

    Parameters:
        project_id (str): Google Cloud project ID for BigQuery.
        patients_limit (int): Maximum number of patients to include in the dataset (0 means no limit; defaults to 20 000 if not specified).
        saving_data (bool): Whether to persist processed tables to disk as .H5 files (default: True).
        out_path (str): Output directory to save HDF5 files (default: "data").
        cache_dir (str): Directory used for caching dataset.
        tables_limit (list): List of table names to include in the dataset (default: common MIMIC-IV tables).
        drop_columns_per_table (dict): Dictionary specifying columns to drop per table (default: predefined subset per table).
        min_age (int): Minimum patient age in years to include (default: 15).
        min_dur (int): Minimum ICU stay duration in hours (default: 36).
        max_dur (int): Maximum ICU stay duration in hours (default: 240).
        location (str): Location for BigQuery client (default: 'US').
        dataset_name (str): Name of the BigQuery dataset (default: 'physionet-data').

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

    def __init__(
        self,
        project_id: str = None,
        patients_limit: int = -1,
        saving_data: bool = True,
        out_path: str = "data",
        cache_dir: str = None,
        tables_limit: list = None,
        drop_columns_per_table=None,
        min_age: int = 15,
        min_dur: int = 36,
        max_dur: int = 240,
        location: str = "US",
        dataset_name: str = "physionet-data",
    ):
        # Lazy import to avoid requiring google-cloud-bigquery for other datasets
        from google.cloud import bigquery

        super().__init__(cache_dir=cache_dir)

        # Load environment variables from .env file (lazy import for optional dependency)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # dotenv is optional - if not available, env vars can still be set manually
            pass

        # Use environment variable if project_id not provided
        if project_id is None:
            try:
                project_id = os.getenv("PROJECT_ID")
            except:
                raise ValueError(
                    "project_id is required: set MIMIC_BQ_PROJECT_ID environment variable (i.e. export MIMIC_BQ_PROJECT_ID='your-project_id')"
                )

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
            patients_limit = 20000
            print(f"patients_limit not provided setting to default {patients_limit}.")

        self.query_args = [
            bigquery.ScalarQueryParameter("limit", "INT64", patients_limit),
            bigquery.ScalarQueryParameter("min_age", "INT64", min_age),
            bigquery.ScalarQueryParameter("min_dur", "INT64", min_dur),
            bigquery.ScalarQueryParameter("max_dur", "INT64", max_dur),
            bigquery.ScalarQueryParameter("min_day", "FLOAT64", float(min_dur) / 24),
        ]

        self.saving_data = saving_data
        self.patients_limit = patients_limit
        self.drop_columns_per_table = drop_columns_per_table

        self.tables_limit = tables_limit

        # If a cached RelBench copy exists (e.g., downloaded db.zip), reuse it and
        # avoid hitting BigQuery.
        cached_db_dir = Path(cache_dir) / "db" if cache_dir is not None else None
        if cached_db_dir and cached_db_dir.exists() and any(cached_db_dir.iterdir()):
            print(
                f"Found cached MIMIC database at {cached_db_dir}, skipping BigQuery build."
            )
            cached_db = Database.load(cached_db_dir)
            self._set_timestamps_from_icustays(cached_db.table_dict["icustays"].df)
            return

        # Create the output directory if it doesn't exist
        current_dir = os.getcwd()
        self.out_path = Path(current_dir) / out_path / f"limit_{patients_limit}"
        if saving_data:
            self.out_path.mkdir(parents=True, exist_ok=True)
            print("Data will be saved to", self.out_path)

        self.client = bigquery.Client(project=project_id, location=location)
        self.dataset_name = dataset_name

        # Set the test and validation timestamps
        # this will be updated based on patients' ICU admission times
        self.test_timestamp = None
        self.val_timestamp = None

        print("Making database in __init__....")
        self.make_db()

    def _set_timestamps_from_icustays(self, icustays_df: pd.DataFrame) -> None:
        timestamps = (
            pd.to_datetime(icustays_df["intime"], errors="coerce")
            .dropna()
            .sort_values()
        )
        if len(timestamps) == 0:
            raise ValueError("Unable to set timestamps: icustays.intime is empty.")

        val_idx = int(len(timestamps) * 0.7)
        test_idx = int(len(timestamps) * 0.85)
        self.val_timestamp = timestamps.iloc[val_idx]
        self.test_timestamp = timestamps.iloc[test_idx]

    def make_db(self) -> Database:
        from google.cloud import bigquery

        start_time = time.time()
        tables_df = {}
        # while MIMIC IV dataset on BigQuery does not have primary and foreign keys set up properly (at all)
        # we need to declare the relations statically here..
        table_names_schemas = {
            "mimiciv_3_1_hosp": [["patients"], ["admissions"]],
            "mimiciv_3_1_icu": [
                ["icustays"],
                ["procedureevents"],
                ["d_items"],
                ["chartevents"],
            ],
        }
        table_key_map = {
            "patients": {"pkey_col": "subject_id", "fkey_col_to_pkey_table": {}},
            "icustays": {
                "pkey_col": "stay_id",
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "subject_id": "patients",
                },
            },
            "admissions": {
                "pkey_col": "hadm_id",
                "fkey_col_to_pkey_table": {"subject_id": "patients"},
            },
            "procedureevents": {
                "pkey_col": "orderid",
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "itemid": "d_items",
                    "stay_id": "icustays",
                    "subject_id": "patients",
                },
            },
            "d_items": {"pkey_col": "itemid", "fkey_col_to_pkey_table": {}},
            "chartevents": {
                "pkey_col": None,
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "itemid": "d_items",
                    "stay_id": "icustays",
                    "subject_id": "patients",
                },
            },
        }

        tables_df["patients"] = Table(
            df=self.get_patients(),
            fkey_col_to_pkey_table=table_key_map["patients"]["fkey_col_to_pkey_table"],
            pkey_col=table_key_map["patients"]["pkey_col"],
            time_col=None,
        )

        query_params = [
            bigquery.ArrayQueryParameter(
                "subject_ids",
                "INT64",
                format_ids(tables_df["patients"].df["subject_id"]),
            ),
            bigquery.ArrayQueryParameter(
                "stay_ids", "INT64", format_ids(tables_df["patients"].df["stay_id"])
            ),
            bigquery.ArrayQueryParameter(
                "hadm_ids", "INT64", format_ids(tables_df["patients"].df["hadm_id"])
            ),
        ]

        tables_df["icustays"] = Table(
            df=self.get_icustays(query_params),
            fkey_col_to_pkey_table=table_key_map["icustays"]["fkey_col_to_pkey_table"],
            pkey_col=table_key_map["icustays"]["pkey_col"],
            time_col="intime",
        )
        query_params = [
            bigquery.ArrayQueryParameter(
                "subject_ids",
                "INT64",
                format_ids(tables_df["icustays"].df["subject_id"]),
            ),
            bigquery.ArrayQueryParameter(
                "stay_ids", "INT64", format_ids(tables_df["icustays"].df["stay_id"])
            ),
            bigquery.ArrayQueryParameter(
                "hadm_ids", "INT64", format_ids(tables_df["icustays"].df["hadm_id"])
            ),
        ]

        for schema, tables in table_names_schemas.items():
            for (table_name,) in tables:
                if table_name in {"patients", "icustays"}:
                    continue
                h5_path = self.out_path / f"{table_name}_{self.patients_limit}.H5"
                if not h5_path.exists():
                    print(f"Creating {table_name}")

                    # Create the query string
                    query_string = self.build_query(table_name, schema)

                    print("Querying table:", table_name, end=" ")
                    # Execute the query
                    df = self.query(
                        query_string=query_string, query_params=query_params
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
                    fkey_col_to_pkey_table=table_key_map[table_name][
                        "fkey_col_to_pkey_table"
                    ],
                    pkey_col=table_key_map[table_name]["pkey_col"],
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
        patients_query = f"""
        SELECT
            i.subject_id,                  -- Unique patient identifier
            i.hadm_id,                     -- Unique hospital admission ID
            i.stay_id,                     -- Unique ICU stay ID
            i.gender,                      -- Patient gender
            ROUND(i.admission_age) AS age, -- Age of the patient at hospital admission
            i.race                         -- Patient race/ethnicity
        FROM `{self.dataset_name}.mimiciv_3_1_derived.icustay_detail` i
        WHERE i.hadm_id IS NOT NULL
           AND i.stay_id IS NOT NULL
           AND i.hospstay_seq = 1
           AND i.icustay_seq = 1
           AND i.los_icu IS NOT NULL
           AND i.admission_age >= @min_age
           AND i.los_icu >= @min_day
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) > (@min_dur*3600)
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) < (@max_dur*3600)
           AND EXISTS (
             SELECT 1
             FROM `{self.dataset_name}.mimiciv_3_1_icu.icustays` icu
             WHERE icu.stay_id = i.stay_id
             )
        QUALIFY
            ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.icustay_seq DESC) = 1
        ORDER BY subject_id
        """

        if self.patients_limit > 0:
            patients_query += "LIMIT @limit"

        H5_fpath = os.path.join(self.out_path, patients_filename)
        if not os.path.exists(H5_fpath):
            patients = self.query(
                query_string=patients_query, query_params=self.query_args
            )
            if self.saving_data:
                patients.to_hdf(H5_fpath, key="patients", index=False)
                print(f"Patients data saved to {H5_fpath}")
        else:
            print(f"File {patients_filename} already exists. Skipping...")
            patients = pd.read_hdf(H5_fpath, key="patients")
        return patients

    def get_icustays(self, params: list):
        icustays_filename = "icustays_" + str(self.patients_limit) + ".H5"
        H5_fpath = os.path.join(self.out_path, icustays_filename)
        # Query the ICU_Stay
        icustays_query = f"""
         SELECT icu.subject_id,
                icu.hadm_id,
                icu.stay_id,
                icu.intime,
                icu.first_careunit,
                icu_detail.los_icu
         FROM `{self.dataset_name}.mimiciv_3_1_icu.icustays` AS icu
              LEFT JOIN `{self.dataset_name}.mimiciv_3_1_derived.icustay_detail` AS icu_detail
              ON icu.stay_id = icu_detail.stay_id
         WHERE icu.subject_id IN UNNEST(@subject_ids)
            AND icu.stay_id IN UNNEST(@stay_ids)
            AND icu.hadm_id IN UNNEST(@hadm_ids)
            AND icu_detail.los_icu IS NOT NULL
         ORDER BY icu.subject_id
         """
        if not os.path.exists(H5_fpath):
            icustays = self.query(query_string=icustays_query, query_params=params)
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
            conditions.append("subject_id IN UNNEST(@subject_ids)")
        if "stay_id" in columns:
            conditions.append("stay_id IN UNNEST(@stay_ids)")
        if "hadm_id" in columns:
            conditions.append("hadm_id IN UNNEST(@hadm_ids)")

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""

        query_string = (
            f"SELECT * FROM `{self.dataset_name}.{schema}.{table_name}`{where_clause};"
        )
        return query_string

    def drop_columns(self, table_name, df):
        drop_cols = list(
            set(self.drop_columns_per_table[table_name]) & set(df.columns)
        )  # intersection
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"Dropped columns from {table_name}: {drop_cols}")
        return df

    def get_columns(self, table_name, schema):
        table_id = f"{self.dataset_name}.{schema}.{table_name}"
        table = self.client.get_table(table_id)

        # Extract column names from the schema
        column_names = [field.name for field in table.schema]
        return column_names

    def query(self, query_string, query_params: list = []):
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        df = self.client.query_and_wait(
            query_string, job_config=job_config
        ).to_dataframe()
        # for writing HDF5 files, we need simple types

        return convert_dtypes(df)


if __name__ == "__main__":
    dataset = MimicDataset()
    db = dataset.make_db()
    mimic_db = dataset.get_db()
    print(mimic_db)

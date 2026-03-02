from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from relbench.base import Database, Dataset, Table


@dataclass(frozen=True)
class TabArenaDatasetSpec:
    slug: str
    name: str
    task_id: int
    dataset_id: int
    target: str
    task_type: str
    num_classes: int
    fold_count: int


TABARENA_DATASETS: dict[str, TabArenaDatasetSpec] = {
    "airfoil-self-noise": TabArenaDatasetSpec(
        slug="airfoil-self-noise",
        name="airfoil_self_noise",
        task_id=363612,
        dataset_id=46904,
        target="scaled-sound-pressure",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=30,
    ),
    "amazon-employee-access": TabArenaDatasetSpec(
        slug="amazon-employee-access",
        name="Amazon_employee_access",
        task_id=363613,
        dataset_id=46905,
        target="ResourceApproved",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "anneal": TabArenaDatasetSpec(
        slug="anneal",
        name="anneal",
        task_id=363614,
        dataset_id=46906,
        target="classes",
        task_type="Supervised Classification",
        num_classes=5,
        fold_count=30,
    ),
    "another-dataset-on-used-fiat-500": TabArenaDatasetSpec(
        slug="another-dataset-on-used-fiat-500",
        name="Another-Dataset-on-used-Fiat-500",
        task_id=363615,
        dataset_id=46907,
        target="price",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=30,
    ),
    "apsfailure": TabArenaDatasetSpec(
        slug="apsfailure",
        name="APSFailure",
        task_id=363616,
        dataset_id=46908,
        target="AirPressureSystemFailure",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "bank-customer-churn": TabArenaDatasetSpec(
        slug="bank-customer-churn",
        name="Bank_Customer_Churn",
        task_id=363619,
        dataset_id=46911,
        target="churn",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "bank-marketing": TabArenaDatasetSpec(
        slug="bank-marketing",
        name="bank-marketing",
        task_id=363618,
        dataset_id=46910,
        target="SubscribeTermDeposit",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "bioresponse": TabArenaDatasetSpec(
        slug="bioresponse",
        name="Bioresponse",
        task_id=363620,
        dataset_id=46912,
        target="MoleculeElicitsResponse",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "blood-transfusion-service-center": TabArenaDatasetSpec(
        slug="blood-transfusion-service-center",
        name="blood-transfusion-service-center",
        task_id=363621,
        dataset_id=46913,
        target="DonatedBloodInMarch2007",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "churn": TabArenaDatasetSpec(
        slug="churn",
        name="churn",
        task_id=363623,
        dataset_id=46915,
        target="CustomerChurned",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "coil2000-insurance-policies": TabArenaDatasetSpec(
        slug="coil2000-insurance-policies",
        name="coil2000_insurance_policies",
        task_id=363624,
        dataset_id=46916,
        target="MobileHomePolicy",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "concrete-compressive-strength": TabArenaDatasetSpec(
        slug="concrete-compressive-strength",
        name="concrete_compressive_strength",
        task_id=363625,
        dataset_id=46917,
        target="ConcreteCompressiveStrength",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=30,
    ),
    "credit-card-clients-default": TabArenaDatasetSpec(
        slug="credit-card-clients-default",
        name="credit_card_clients_default",
        task_id=363627,
        dataset_id=46919,
        target="DefaultOnPaymentNextMonth",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "credit-g": TabArenaDatasetSpec(
        slug="credit-g",
        name="credit-g",
        task_id=363626,
        dataset_id=46918,
        target="good_or_bad_customer",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "customer-satisfaction-in-airline": TabArenaDatasetSpec(
        slug="customer-satisfaction-in-airline",
        name="customer_satisfaction_in_airline",
        task_id=363628,
        dataset_id=46920,
        target="satisfaction",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "diabetes": TabArenaDatasetSpec(
        slug="diabetes",
        name="diabetes",
        task_id=363629,
        dataset_id=46921,
        target="TestedPositiveForDiabetes",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "diabetes130us": TabArenaDatasetSpec(
        slug="diabetes130us",
        name="Diabetes130US",
        task_id=363630,
        dataset_id=46922,
        target="EarlyReadmission",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "diamonds": TabArenaDatasetSpec(
        slug="diamonds",
        name="diamonds",
        task_id=363631,
        dataset_id=46923,
        target="price",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "e-commereshippingdata": TabArenaDatasetSpec(
        slug="e-commereshippingdata",
        name="E-CommereShippingData",
        task_id=363632,
        dataset_id=46924,
        target="ArrivedLate",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "fitness-club": TabArenaDatasetSpec(
        slug="fitness-club",
        name="Fitness_Club",
        task_id=363671,
        dataset_id=46927,
        target="attended",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "food-delivery-time": TabArenaDatasetSpec(
        slug="food-delivery-time",
        name="Food_Delivery_Time",
        task_id=363672,
        dataset_id=46928,
        target="Time_taken(min)",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "givemesomecredit": TabArenaDatasetSpec(
        slug="givemesomecredit",
        name="GiveMeSomeCredit",
        task_id=363673,
        dataset_id=46929,
        target="FinancialDistressNextTwoYears",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "hazelnut-spread-contaminant-detection": TabArenaDatasetSpec(
        slug="hazelnut-spread-contaminant-detection",
        name="hazelnut-spread-contaminant-detection",
        task_id=363674,
        dataset_id=46930,
        target="Contaminated",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "healthcare-insurance-expenses": TabArenaDatasetSpec(
        slug="healthcare-insurance-expenses",
        name="healthcare_insurance_expenses",
        task_id=363675,
        dataset_id=46931,
        target="charges",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=30,
    ),
    "heloc": TabArenaDatasetSpec(
        slug="heloc",
        name="heloc",
        task_id=363676,
        dataset_id=46932,
        target="RiskPerformance",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "hiva-agnostic": TabArenaDatasetSpec(
        slug="hiva-agnostic",
        name="hiva_agnostic",
        task_id=363677,
        dataset_id=46933,
        target="CompoundActivity",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=9,
    ),
    "houses": TabArenaDatasetSpec(
        slug="houses",
        name="houses",
        task_id=363678,
        dataset_id=46934,
        target="LnMedianHouseValue",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "hr-analytics-job-change-of-data-scientists": TabArenaDatasetSpec(
        slug="hr-analytics-job-change-of-data-scientists",
        name="HR_Analytics_Job_Change_of_Data_Scientists",
        task_id=363679,
        dataset_id=46935,
        target="LookingForJobChange",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "in-vehicle-coupon-recommendation": TabArenaDatasetSpec(
        slug="in-vehicle-coupon-recommendation",
        name="in_vehicle_coupon_recommendation",
        task_id=363681,
        dataset_id=46937,
        target="AcceptCoupon",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "is-this-a-good-customer": TabArenaDatasetSpec(
        slug="is-this-a-good-customer",
        name="Is-this-a-good-customer",
        task_id=363682,
        dataset_id=46938,
        target="bad_client_target",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "jm1": TabArenaDatasetSpec(
        slug="jm1",
        name="jm1",
        task_id=363712,
        dataset_id=46979,
        target="defects",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "kddcup09-appetency": TabArenaDatasetSpec(
        slug="kddcup09-appetency",
        name="kddcup09_appetency",
        task_id=363683,
        dataset_id=46939,
        target="appetency",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "marketing-campaign": TabArenaDatasetSpec(
        slug="marketing-campaign",
        name="Marketing_Campaign",
        task_id=363684,
        dataset_id=46940,
        target="Response",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "maternal-health-risk": TabArenaDatasetSpec(
        slug="maternal-health-risk",
        name="maternal_health_risk",
        task_id=363685,
        dataset_id=46941,
        target="RiskLevel",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=30,
    ),
    "miami-housing": TabArenaDatasetSpec(
        slug="miami-housing",
        name="miami_housing",
        task_id=363686,
        dataset_id=46942,
        target="SALE_PRC",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "mic": TabArenaDatasetSpec(
        slug="mic",
        name="MIC",
        task_id=363711,
        dataset_id=46980,
        target="LET_IS",
        task_type="Supervised Classification",
        num_classes=8,
        fold_count=30,
    ),
    "naticusdroid": TabArenaDatasetSpec(
        slug="naticusdroid",
        name="NATICUSdroid",
        task_id=363689,
        dataset_id=46969,
        target="Malware",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "online-shoppers-intention": TabArenaDatasetSpec(
        slug="online-shoppers-intention",
        name="online_shoppers_intention",
        task_id=363691,
        dataset_id=46947,
        target="Revenue",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "physiochemical-protein": TabArenaDatasetSpec(
        slug="physiochemical-protein",
        name="physiochemical_protein",
        task_id=363693,
        dataset_id=46949,
        target="ResidualSize",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "polish-companies-bankruptcy": TabArenaDatasetSpec(
        slug="polish-companies-bankruptcy",
        name="polish_companies_bankruptcy",
        task_id=363694,
        dataset_id=46950,
        target="company_bankrupt",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "qsar-biodeg": TabArenaDatasetSpec(
        slug="qsar-biodeg",
        name="qsar-biodeg",
        task_id=363696,
        dataset_id=46952,
        target="Biodegradable",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=30,
    ),
    "qsar-fish-toxicity": TabArenaDatasetSpec(
        slug="qsar-fish-toxicity",
        name="QSAR_fish_toxicity",
        task_id=363698,
        dataset_id=46954,
        target="LC50",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=30,
    ),
    "qsar-tid-11": TabArenaDatasetSpec(
        slug="qsar-tid-11",
        name="QSAR-TID-11",
        task_id=363697,
        dataset_id=46953,
        target="MEDIAN_PXC50",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "sdss17": TabArenaDatasetSpec(
        slug="sdss17",
        name="SDSS17",
        task_id=363699,
        dataset_id=46955,
        target="ObjectType",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=9,
    ),
    "seismic-bumps": TabArenaDatasetSpec(
        slug="seismic-bumps",
        name="seismic-bumps",
        task_id=363700,
        dataset_id=46956,
        target="HighEnergySeismicBump",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "splice": TabArenaDatasetSpec(
        slug="splice",
        name="splice",
        task_id=363702,
        dataset_id=46958,
        target="SiteType",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=9,
    ),
    "students-dropout-and-academic-success": TabArenaDatasetSpec(
        slug="students-dropout-and-academic-success",
        name="students_dropout_and_academic_success",
        task_id=363704,
        dataset_id=46960,
        target="AcademicOutcome",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=9,
    ),
    "superconductivity": TabArenaDatasetSpec(
        slug="superconductivity",
        name="superconductivity",
        task_id=363705,
        dataset_id=46961,
        target="critical_temp",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
    "taiwanese-bankruptcy-prediction": TabArenaDatasetSpec(
        slug="taiwanese-bankruptcy-prediction",
        name="taiwanese_bankruptcy_prediction",
        task_id=363706,
        dataset_id=46962,
        target="Bankrupt",
        task_type="Supervised Classification",
        num_classes=2,
        fold_count=9,
    ),
    "website-phishing": TabArenaDatasetSpec(
        slug="website-phishing",
        name="website_phishing",
        task_id=363707,
        dataset_id=46963,
        target="WebsiteType",
        task_type="Supervised Classification",
        num_classes=3,
        fold_count=30,
    ),
    "wine-quality": TabArenaDatasetSpec(
        slug="wine-quality",
        name="wine_quality",
        task_id=363708,
        dataset_id=46964,
        target="median_wine_quality",
        task_type="Supervised Regression",
        num_classes=0,
        fold_count=9,
    ),
}


def get_tabarena_dataset_slugs() -> list[str]:
    return sorted(TABARENA_DATASETS.keys())


def _import_openml():
    try:
        import openml
    except ImportError as exc:
        raise ImportError(
            "TabArena datasets require the `openml` package. Install it with "
            "`pip install relbench[tabarena]` (or `pip install openml`)."
        ) from exc
    return openml


def _problem_type_from_spec(task_type: str, num_classes: int) -> str:
    if task_type == "Supervised Regression":
        return "regression"
    if int(num_classes) <= 2:
        return "binary"
    return "multiclass"


class TabArenaDataset(Dataset):
    r"""Single-table RelBench dataset wrapper over TabArena OpenML tasks."""

    url = "https://huggingface.co/datasets/TabArena/benchmark_results"
    val_timestamp = pd.Timestamp("2000-01-02")
    test_timestamp = pd.Timestamp("2000-01-03")

    def __init__(self, *, dataset_slug: str, cache_dir: Optional[str] = None):
        slug = str(dataset_slug)
        if slug not in TABARENA_DATASETS:
            raise ValueError(
                f"Unknown TabArena dataset slug={slug!r}. Known values: {sorted(TABARENA_DATASETS.keys())}"
            )

        self.spec = TABARENA_DATASETS[slug]
        self.name = f"tabarena-{slug}"
        self.tabarena_name = self.spec.name
        self.task_id = int(self.spec.task_id)
        self.openml_dataset_id = int(self.spec.dataset_id)
        self.target_name = self.spec.target
        self.problem_type = _problem_type_from_spec(
            self.spec.task_type, self.spec.num_classes
        )
        self.num_classes = (
            int(self.spec.num_classes) if self.problem_type != "regression" else 0
        )

        self._openml_task = None
        self._X_df: Optional[pd.DataFrame] = None
        self._y_encoded: Optional[np.ndarray] = None

        super().__init__(cache_dir=cache_dir)

    @property
    def available_folds(self) -> list[int]:
        return list(range(int(self.spec.fold_count)))

    def _load_task_with_retry(self, task_id: int, retries: int = 4):
        openml = _import_openml()
        delay = 1.0
        for attempt in range(retries + 1):
            try:
                return openml.tasks.get_task(
                    task_id,
                    download_splits=True,
                    download_data=True,
                    download_qualities=False,
                    download_features_meta_data=True,
                )
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2.0

    def _ensure_openml_loaded(self) -> None:
        if (
            self._openml_task is not None
            and self._X_df is not None
            and self._y_encoded is not None
        ):
            return

        task = self._load_task_with_retry(self.task_id)
        X_df, y_ser, _cat, _names = task.get_dataset().get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )

        X_df = pd.DataFrame(X_df).reset_index(drop=True)
        y_ser = pd.Series(y_ser, name=task.target_name).reset_index(drop=True)

        if self.problem_type == "regression":
            y_encoded = y_ser.astype(float).to_numpy(copy=True)
        else:
            cat = pd.Categorical(y_ser)
            if cat.codes.min() < 0:
                raise RuntimeError(
                    f"Encountered missing labels in OpenML task_id={self.task_id} ({self.tabarena_name})."
                )
            y_encoded = cat.codes.astype(np.int64, copy=False)
            detected_num_classes = int(len(cat.categories))
            if self.num_classes and detected_num_classes != self.num_classes:
                raise RuntimeError(
                    f"Label cardinality mismatch for {self.tabarena_name}: expected {self.num_classes}, got {detected_num_classes}."
                )
            self.num_classes = detected_num_classes

        if len(X_df) != len(y_encoded):
            raise RuntimeError(
                f"Feature/label row mismatch for {self.tabarena_name}: {len(X_df)} vs {len(y_encoded)}."
            )

        self._openml_task = task
        self._X_df = X_df
        self._y_encoded = y_encoded

    def get_openml_task(self):
        self._ensure_openml_loaded()
        return self._openml_task

    def get_target_array(self) -> np.ndarray:
        self._ensure_openml_loaded()
        assert self._y_encoded is not None
        return self._y_encoded

    def get_openml_fold_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        fold = int(fold)
        if fold < 0 or fold >= int(self.spec.fold_count):
            raise ValueError(
                f"Invalid fold={fold} for {self.name}. Valid folds are 0..{int(self.spec.fold_count) - 1}."
            )

        task = self.get_openml_task()
        n_repeats, n_folds, _n_samples = task.get_split_dimensions()
        repeat = fold // int(n_folds)
        fold_in_repeat = fold % int(n_folds)
        if repeat >= int(n_repeats):
            raise ValueError(
                f"Fold index {fold} exceeds OpenML split dimensions for {self.name}: repeats={n_repeats}, folds={n_folds}."
            )

        train_idx, test_idx = task.get_train_test_split_indices(
            repeat=repeat,
            fold=fold_in_repeat,
            sample=0,
        )
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    def make_db(self) -> Database:
        self._ensure_openml_loaded()
        assert self._X_df is not None

        records = self._X_df.copy(deep=True)
        records.insert(0, "record_id", np.arange(len(records), dtype=np.int64))

        return Database(
            {
                "records": Table(
                    df=records,
                    fkey_col_to_pkey_table={},
                    pkey_col="record_id",
                    time_col=None,
                )
            }
        )

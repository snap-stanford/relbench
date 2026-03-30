"""Validate the TabArena RelBench wrapper with a public baseline.

The script trains a baseline on the original OpenML rows selected by a RelBench
task split, then runs inference on both the original and relbenchified views of
the same validation and test rows. Matching predictions and metrics confirm that
the RelBench wrapper preserves the original single-table task semantics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from relbench.datasets.tabarena import TabArenaDataset, get_tabarena_dataset_slugs
from relbench.tasks.tabarena import TabArenaSplitEntityTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="credit-g",
        choices=get_tabarena_dataset_slugs(),
        help="TabArena dataset slug, for example `credit-g` or `airfoil-self-noise`.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="OpenML split index exposed in RelBench as `split-<index>`.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random-forest",
        choices=["random-forest", "xgboost"],
        help="Public baseline model used for validation.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
    )
    return parser.parse_args()


def _load_openml_frame(dataset: TabArenaDataset) -> pd.DataFrame:
    task = dataset.get_openml_task()
    X_df, y_ser, _cat, _names = task.get_dataset().get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    X_df = pd.DataFrame(X_df).reset_index(drop=True)
    y_ser = pd.Series(y_ser, name=task.target_name).reset_index(drop=True)
    _ = y_ser
    return X_df


def _join_records(records_df: pd.DataFrame, task_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    joined = task_df.merge(records_df, on="record_id", how="left", validate="1:1")
    X_df = joined.drop(columns=["record_id", "target"])
    y = joined["target"].to_numpy()
    return X_df, y


def _build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        col
        for col in X_df.columns
        if pd.api.types.is_object_dtype(X_df[col])
        or isinstance(X_df[col].dtype, CategoricalDtype)
        or pd.api.types.is_bool_dtype(X_df[col])
    ]
    numeric_cols = [col for col in X_df.columns if col not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _build_model(
    *,
    task: TabArenaSplitEntityTask,
    X_df: pd.DataFrame,
    model_name: str,
    random_state: int,
) -> Pipeline:
    preprocessor = _build_preprocessor(X_df)

    if model_name == "random-forest":
        if task.task_type.value == "regression":
            estimator = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise ImportError(
                "The `xgboost` example requires `pip install xgboost`."
            ) from exc

        if task.task_type.value == "regression":
            estimator = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=random_state,
            )
        else:
            estimator = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=random_state,
            )
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported model {model_name!r}")

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def _predict(
    model: Pipeline,
    task: TabArenaSplitEntityTask,
    X_df: pd.DataFrame,
) -> np.ndarray:
    if task.task_type.value == "regression":
        return np.asarray(model.predict(X_df), dtype=np.float64)
    proba = model.predict_proba(X_df)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError(f"Expected binary predict_proba output, got shape={proba.shape}")
    return np.asarray(proba[:, 1], dtype=np.float64)


def _metric_name_and_value(
    task: TabArenaSplitEntityTask, y_true: np.ndarray, pred: np.ndarray
) -> tuple[str, float]:
    if task.task_type.value == "regression":
        return "rmse", float(np.sqrt(mean_squared_error(y_true, pred)))
    return "auroc", float(roc_auc_score(y_true, pred))


def _max_prediction_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(a - b))) if len(a) else 0.0


def main() -> None:
    args = parse_args()

    dataset = TabArenaDataset(dataset_slug=args.dataset)
    task = TabArenaSplitEntityTask(dataset, split=args.split)
    records_df = dataset.get_db().table_dict["records"].df.reset_index(drop=True)
    openml_X = _load_openml_frame(dataset)
    openml_y = pd.Series(dataset.get_target_array(), name="target").reset_index(drop=True)

    relbench_train = task.get_table("train", mask_input_cols=False).df
    relbench_val = task.get_table("val", mask_input_cols=False).df
    relbench_test = task.get_table("test", mask_input_cols=False).df

    X_train_rb, y_train_rb = _join_records(records_df, relbench_train)
    X_val_rb, y_val_rb = _join_records(records_df, relbench_val)
    X_test_rb, y_test_rb = _join_records(records_df, relbench_test)

    X_train_orig = openml_X.iloc[relbench_train["record_id"].to_numpy()].reset_index(drop=True)
    y_train_orig = openml_y.iloc[relbench_train["record_id"].to_numpy()].to_numpy()
    X_val_orig = openml_X.iloc[relbench_val["record_id"].to_numpy()].reset_index(drop=True)
    y_val_orig = openml_y.iloc[relbench_val["record_id"].to_numpy()].to_numpy()
    X_test_orig = openml_X.iloc[relbench_test["record_id"].to_numpy()].reset_index(drop=True)
    y_test_orig = openml_y.iloc[relbench_test["record_id"].to_numpy()].to_numpy()

    print("[data equality checks]")
    print(f"train features identical: {X_train_orig.equals(X_train_rb)}")
    print(f"val features identical:   {X_val_orig.equals(X_val_rb)}")
    print(f"test features identical:  {X_test_orig.equals(X_test_rb)}")
    print(f"train labels identical:   {np.array_equal(y_train_orig, y_train_rb)}")
    print(f"val labels identical:     {np.array_equal(y_val_orig, y_val_rb)}")
    print(f"test labels identical:    {np.array_equal(y_test_orig, y_test_rb)}")
    print()

    model = _build_model(
        task=task,
        X_df=X_train_orig,
        model_name=args.model,
        random_state=args.random_state,
    )
    model.fit(X_train_orig, y_train_orig)

    val_pred_orig = _predict(model, task, X_val_orig)
    val_pred_rb = _predict(model, task, X_val_rb)
    test_pred_orig = _predict(model, task, X_test_orig)
    test_pred_rb = _predict(model, task, X_test_rb)

    metric_name, val_metric_orig = _metric_name_and_value(task, y_val_orig, val_pred_orig)
    _, val_metric_rb = _metric_name_and_value(task, y_val_rb, val_pred_rb)
    _, test_metric_orig = _metric_name_and_value(task, y_test_orig, test_pred_orig)
    _, test_metric_rb = _metric_name_and_value(task, y_test_rb, test_pred_rb)

    print("[prediction consistency]")
    print(f"val max |orig - relbench|:  {_max_prediction_delta(val_pred_orig, val_pred_rb):.6g}")
    print(f"test max |orig - relbench|: {_max_prediction_delta(test_pred_orig, test_pred_rb):.6g}")
    print()

    print(f"[{metric_name}]")
    print(f"validation original-openml: {val_metric_orig:.6f}")
    print(f"validation relbenchified:   {val_metric_rb:.6f}")
    print(f"test original-openml:       {test_metric_orig:.6f}")
    print(f"test relbenchified:         {test_metric_rb:.6f}")


if __name__ == "__main__":
    main()

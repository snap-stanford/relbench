"""Inspect how a TabArena OpenML task is represented in RelBench.

This script compares the original OpenML dataset/task with the RelBench wrapper:
the source rows become a single ``records`` table, while each ``split-*`` task
materializes a thin task table keyed by ``record_id``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

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
        "--show_rows",
        type=int,
        default=3,
        help="Number of joined examples to print from the RelBench train split.",
    )
    return parser.parse_args()


def _load_openml_frame(dataset: TabArenaDataset) -> tuple[pd.DataFrame, pd.Series]:
    task = dataset.get_openml_task()
    X_df, y_ser, _cat, _names = task.get_dataset().get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    X_df = pd.DataFrame(X_df).reset_index(drop=True)
    y_ser = pd.Series(y_ser, name=task.target_name).reset_index(drop=True)
    return X_df, y_ser


def _join_records(records_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    joined = task_df.merge(records_df, on="record_id", how="left", validate="1:1")
    feature_cols = [col for col in joined.columns if col not in {"record_id", "target"}]
    return joined[["record_id", *feature_cols, "target"]]


def _check_translation(
    dataset: TabArenaDataset,
    task: TabArenaSplitEntityTask,
) -> dict[str, object]:
    X_df, _y_ser = _load_openml_frame(dataset)
    y_encoded = pd.Series(dataset.get_target_array(), name="target").reset_index(drop=True)
    records_df = dataset.get_db().table_dict["records"].df.reset_index(drop=True)

    openml_train_idx, openml_test_idx = dataset.get_openml_split_indices(task.split)
    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)

    relbench_train_ids = train_table.df["record_id"].to_numpy()
    relbench_val_ids = val_table.df["record_id"].to_numpy()
    relbench_test_ids = test_table.df["record_id"].to_numpy()
    relbench_trainval_ids = set(relbench_train_ids).union(relbench_val_ids)

    target_matches = True
    for split_name, split_df in {
        "train": train_table.df,
        "val": val_table.df,
        "test": test_table.df,
    }.items():
        relbench_target = split_df["target"].reset_index(drop=True)
        source_target = y_encoded.iloc[split_df["record_id"].to_numpy()].reset_index(drop=True)
        if not relbench_target.equals(source_target):
            target_matches = False
            print(f"[check] target mismatch on {split_name}")

    return {
        "dataset_slug": dataset.spec.slug,
        "dataset_name": dataset.name,
        "tabarena_name": dataset.tabarena_name,
        "problem_type": dataset.problem_type,
        "openml_task_id": dataset.task_id,
        "openml_dataset_id": dataset.openml_dataset_id,
        "target_name": dataset.target_name,
        "records_rows": len(records_df),
        "records_feature_columns": len(records_df.columns) - 1,
        "openml_rows": len(X_df),
        "openml_train_rows": len(openml_train_idx),
        "openml_test_rows": len(openml_test_idx),
        "relbench_train_rows": len(train_table.df),
        "relbench_val_rows": len(val_table.df),
        "relbench_test_rows": len(test_table.df),
        "records_match_openml_rows": len(records_df) == len(X_df),
        "record_ids_are_row_indices": np.array_equal(
            records_df["record_id"].to_numpy(),
            np.arange(len(records_df), dtype=np.int64),
        ),
        "relbench_test_matches_openml_test": set(relbench_test_ids) == set(openml_test_idx),
        "relbench_train_val_partition_openml_train": relbench_trainval_ids == set(
            openml_train_idx
        ),
        "relbench_train_val_are_disjoint": set(relbench_train_ids).isdisjoint(
            relbench_val_ids
        ),
        "targets_match_source_rows": target_matches,
    }


def main() -> None:
    args = parse_args()

    dataset = TabArenaDataset(dataset_slug=args.dataset)
    task = TabArenaSplitEntityTask(dataset, split=args.split)
    records_df = dataset.get_db().table_dict["records"].df.reset_index(drop=True)

    print("[dataset spec]")
    print(asdict(dataset.spec))
    print()

    summary = _check_translation(dataset, task)
    print("[translation summary]")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()

    train_table = task.get_table("train", mask_input_cols=False)
    joined_train = _join_records(records_df, train_table.df).head(args.show_rows)
    print("[joined relbench train rows]")
    print(joined_train.to_string(index=False))


if __name__ == "__main__":
    main()

"""Utilities to inspect how TabArena datasets are represented in RelBench."""

import argparse
from pathlib import Path

import pandas as pd

from relbench.datasets import get_dataset
from relbench.datasets.tabarena import TABARENA_DATASETS, get_tabarena_dataset_slugs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_slugs",
        type=str,
        default="all",
        help=(
            "Comma-separated TabArena slugs or 'all'. "
            "Example: credit-g,airfoil-self-noise"
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/tabarena_relbench_translation.csv",
    )
    parser.add_argument(
        "--include_fold_examples",
        action="store_true",
        default=False,
        help="If set, writes one sample record for fold-0 per dataset.",
    )
    return parser.parse_args()


def _parse_dataset_slugs(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return get_tabarena_dataset_slugs()
    slugs = [slug.strip() for slug in arg.split(",") if slug.strip()]
    valid = set(get_tabarena_dataset_slugs())
    invalid = [slug for slug in slugs if slug not in valid]
    if invalid:
        raise ValueError(f"Unknown dataset slugs: {invalid}")
    return slugs


def _summarize_dataset(dataset_name: str) -> dict:
    dataset = get_dataset(dataset_name, download=False)
    spec = TABARENA_DATASETS[dataset.name.replace("tabarena-", "")]
    db = dataset.make_db()
    records = db.table_dict["records"]
    row = {
        "dataset_slug": spec.slug,
        "dataset_name": dataset.name,
        "tabarena_benchmark_name": spec.name,
        "openml_task_id": spec.task_id,
        "openml_dataset_id": spec.dataset_id,
        "target_col": spec.target,
        "problem_type": spec.task_type,
        "num_classes": spec.num_classes,
        "fold_count": spec.fold_count,
        "records_rows": int(len(records.df)),
        "records_columns": int(len(records.df.columns)),
        "entity_table": "records",
        "entity_pkey": records.pkey_col,
        "split_timestamp_columns": bool(records.time_col is not None),
    }
    return row


def _summarize_fold_example(dataset_name: str) -> list[dict]:
    from relbench.tasks import get_task

    rows: list[dict] = []
    for fold in [0]:
        task = get_task(dataset_name, f"fold-{fold}")
        train = task.get_table("train", mask_input_cols=False)
        val = task.get_table("val", mask_input_cols=False)
        test = task.get_table("test", mask_input_cols=False)
        rows.append(
            {
                "dataset_name": dataset_name,
                "fold": int(fold),
                "task_type": str(task.task_type.value),
                "fold_train_rows": int(len(train)),
                "fold_val_rows": int(len(val)),
                "fold_test_rows": int(len(test)),
                "train_columns": int(len(train.df.columns)),
                "time_col": str(train.time_col),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    dataset_slugs = _parse_dataset_slugs(args.dataset_slugs)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict] = []
    split_rows: list[dict] = []
    for slug in dataset_slugs:
        dataset_name = f"tabarena-{slug}"
        print(f"[Inspect] {dataset_name}")
        dataset_rows.append(_summarize_dataset(dataset_name))
        if args.include_fold_examples:
            split_rows.extend(_summarize_fold_example(dataset_name))

    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_path, index=False)

    if split_rows:
        split_path = output_path.with_name(f"{output_path.stem}_fold_samples.csv")
        pd.DataFrame(split_rows).to_csv(split_path, index=False)
        print(f"[Done] dataset summary: {output_path}")
        print(f"[Done] fold sample summary: {split_path}")
    else:
        print(f"[Done] dataset summary: {output_path}")


if __name__ == "__main__":
    main()

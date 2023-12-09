import argparse
from typing import Dict

import numpy as np
import torch

from relbench.data import RelBenchDataset, Table
from relbench.data.task import TaskType
from relbench.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-engage")
# Classification task: rel-stackex-engage
# Regression task: rel-stackex-votes
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: remove process=True once correct data is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task = dataset.get_task(args.task)

train_table = task.train_table
val_table = task.val_table
test_table = task.test_table


def global_zero(train_table: Table, pred_table: Table) -> Dict[str, float]:
    pred = np.zeros(len(pred_table))
    return task.evaluate(pred, pred_table)


def global_mean(train_table: Table, pred_table: Table) -> Dict[str, float]:
    mean = train_table.df[task.target_col].astype(float).values.mean()
    pred = np.ones(len(pred_table)) * mean

    return task.evaluate(pred, pred_table)


def global_median(train_table: Table, pred_table: Table) -> Dict[str, float]:
    median = np.median(train_table.df[task.target_col].astype(float).values)
    pred = np.ones(len(pred_table)) * median

    return task.evaluate(pred, pred_table)


def entity_mean(train_table: Table, pred_table: Table) -> Dict[str, float]:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
    df = pred_table.df.merge(df, how="left", on=fkey)
    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values

    return task.evaluate(pred, pred_table)


def entity_median(train_table: Table, pred_table: Table) -> Dict[str, float]:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
    df = pred_table.df.merge(df, how="left", on=fkey)
    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values

    return task.evaluate(pred, pred_table)


def random(train_table: Table, pred_table: Table) -> Dict[str, float]:
    pred = np.random.rand(len(pred_table))
    return task.evaluate(pred, pred_table)


def majority(train_table: Table, pred_table: Table) -> Dict[str, float]:
    past_target = train_table.df[task.target_col].astype(int)
    majority_label = int(past_target.mode())
    pred = torch.full((len(pred_table),), fill_value=majority_label)
    return task.evaluate(pred, pred_table)


if task.task_type == TaskType.REGRESSION:
    train_metrics = global_zero(train_table, train_table)
    val_metrics = global_zero(train_table, val_table)
    print("Global Zero:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = global_mean(train_table, train_table)
    val_metrics = global_mean(train_table, val_table)
    print("Global Mean:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = global_median(train_table, train_table)
    val_metrics = global_median(train_table, val_table)
    print("Global Median:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = entity_mean(train_table, train_table)
    val_metrics = entity_mean(train_table, val_table)
    print("Entity Mean:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = entity_median(train_table, train_table)
    val_metrics = entity_median(train_table, val_table)
    print("Entity Median:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    train_metrics = random(train_table, train_table)
    val_metrics = random(train_table, val_table)
    print("Random")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = majority(train_table, train_table)
    val_metrics = majority(train_table, val_table)
    print("Majority:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")
